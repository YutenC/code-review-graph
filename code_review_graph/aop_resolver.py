"""Post-build Spring AOP advice-to-target call resolver.

Tree-sitter parsing sees a ``@Before``/``@Around``/... advice method inside an
``@Aspect`` class and the business methods it intercepts as two unrelated
methods — there is no ``CALLS`` edge between them, because the interception
happens through a Spring AOP proxy at runtime, not through a static method
call in the advice's source text.  This module approximates that missing
edge by reading the pointcut expression each advice annotation carries
(already captured verbatim in ``extra["decorators"]`` by the parser) and
matching it against candidate target methods already in the graph.

Resolution chain:
    ``@Aspect`` class
        → ``@Pointcut``/``@Before``/``@After``/``@Around``/``@AfterReturning``/
          ``@AfterThrowing`` methods inside it
        → pointcut expression (inline, or via a same-class named
          ``@Pointcut`` reference)
        → candidate target methods matched against that expression
        → ``CALLS`` edge, advice → target, tagged ``extra.aop_resolved``

Scope (see issue #592 — Option 1, regex-based approximation):

* ``@annotation(some.pkg.Foo)`` pointcuts are matched by simple annotation
  name (``Foo``) against ``extra["decorators"]`` on the candidate method
  itself. This is exact enough for the common case, but two same-named
  annotations in different packages are indistinguishable — a known
  approximation, consistent with the rest of Option 1. A *class-level*
  ``extra["spring_annotations"]`` match (e.g. a class-level
  ``@Transactional``) is intentionally **not** expanded to every method in
  that class — that is ``@within()`` semantics, a different pointcut
  designator, and conflating the two produced widespread false positives
  on common class-level stereotype annotations (see PR #916 review).
* ``execution(...)`` pointcuts are converted into a regex applied only to
  the ``ClassName.methodName`` portion of a candidate — the return-type
  pattern, parameter pattern (``(..)``), and any package portion of the
  declaring-type pattern are all discarded.  The package portion is dropped
  because parsed nodes are not persisted with a resolved Java package (the
  ``package`` declaration is only available transiently, during parsing of
  that one file); matching therefore is a strict subset of full AspectJ
  semantics and **can produce both false positives and false negatives**.
  This is deliberate and documented, not an oversight — see issue #592.
  When dropping the package portion reduces the pattern to something that
  would match virtually every method (e.g. ``com.foo.service.*.*`` losing
  its package prefix down to a bare ``*.*``), the expression is treated as
  unresolvable rather than emitting a nearly-unfiltered edge set — see
  ``_parse_execution_expression``.
* Pointcut expressions combining multiple sub-expressions with ``&&`` or
  ``||`` are compound boolean pointcuts and are skipped entirely rather
  than partially parsed, to avoid emitting a wrong edge. This check also
  applies to the expression a same-class named ``@Pointcut`` reference
  resolves to, not just the reference text itself. A bare ``!`` negation
  is not recognized by either pointcut-kind matcher below, so it is
  skipped too — just by falling through unparsed rather than via an
  explicit compound-operator check.
* ``within(Type+)``, ``args()``/``target()``/``this()``, and named pointcut
  references that cross an ``@Aspect`` class boundary are out of scope for
  this pass (see the narrower AOP ``LanguageGap`` note in
  ``uncertainty.py``) and are left for a future PR.

Reuses the existing ``CALLS`` edge kind rather than introducing a new one,
so advice→target edges are automatically picked up by ``callers_of``,
``callees_of``, and ``get_impact_radius`` without touching their edge-kind
allowlists. Two dedicated query patterns, ``advises`` and ``advised_by``,
filter this same data down to only the ``extra.aop_resolved`` edges — use
those when the goal is specifically "what AOP relationships exist here"
rather than every caller/callee regardless of provenance.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Optional

from .parser import EdgeInfo

if TYPE_CHECKING:
    from .graph import GraphStore

logger = logging.getLogger(__name__)

_ADVICE_ANNOTATIONS = frozenset(
    {"Before", "After", "Around", "AfterReturning", "AfterThrowing"}
)

# A pointcut expression combining sub-expressions with boolean operators is
# out of scope for this regex-approximation pass — see module docstring.
_COMPOUND_OPERATORS = ("&&", "||")

# A bare same-class named-pointcut reference, e.g. "pointcut1()". A dotted
# reference (e.g. "com.foo.OtherAspect.pointcut1()") is a cross-aspect
# reference, which is explicitly out of scope (see module docstring).
_NAMED_POINTCUT_REF = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\(\s*\)\s*$")
_ANNOTATION_POINTCUT = re.compile(r"^\s*@annotation\s*\((.+)\)\s*$", re.DOTALL)
_EXECUTION_POINTCUT = re.compile(r"^\s*execution\s*\((.*)\)\s*$", re.DOTALL)

_DERIVED_FLAG = "aop_resolved"

# What target_pattern reduces to when every segment collapsed to a bare
# wildcard — i.e. the package portion was the only thing that made the
# original execution() pattern selective. Resolving these would match
# nearly every method in the codebase instead of the intended scope, so
# _parse_execution_expression treats them as unresolvable.
_UNIVERSAL_REGEXES = frozenset({"^[^.]*$", r"^[^.]*\.[^.]*$"})


def _extract_annotation_string_arg(deco_text: str) -> Optional[str]:
    """Return the pointcut expression embedded in an advice annotation's raw text.

    Handles the common positional form (``@Before("expr")``) as well as
    named-argument forms (``@AfterThrowing(pointcut = "expr", throwing =
    "ex")`` / ``value = "expr"``). Returns ``None`` when no string literal
    argument can be found.
    """
    named = re.search(r'(?:pointcut|value)\s*=\s*"((?:[^"\\]|\\.)*)"', deco_text)
    if named:
        return named.group(1)
    any_string = re.search(r'"((?:[^"\\]|\\.)*)"', deco_text)
    if any_string:
        return any_string.group(1)
    return None


def _annotation_head(deco_text: str) -> str:
    """Return the annotation name from a raw decorator string, dropping args."""
    return deco_text.split("(", 1)[0].strip()


def _aspectj_pattern_to_regex(pattern: str) -> str:
    """Convert one AspectJ-style dotted wildcard pattern into a regex fragment.

    Rules (approximate — see module docstring for the scope this is applied
    within):

    * ``..`` means "any number of intervening path segments" → ``.*``
    * a lone ``.`` is a literal separator → escaped ``\\.``
    * ``*`` within a segment means "anything but a dot" → ``[^.]*``
    * every other character is matched literally
    """
    out: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "." and i + 1 < n and pattern[i + 1] == ".":
            out.append(".*")
            i += 2
        elif ch == ".":
            out.append(r"\.")
            i += 1
        elif ch == "*":
            out.append("[^.]*")
            i += 1
        else:
            out.append(re.escape(ch))
            i += 1
    return "".join(out)


def _parse_execution_expression(expr: str) -> Optional[tuple[str, bool]]:
    """Reduce an ``execution(...)`` pointcut expression to a (regex, needs_class) pair.

    ``needs_class`` is True when the expression's method-pattern token
    included a declaring-type segment (e.g. ``com.foo.*.*``), so the regex
    must be matched against ``ClassName.methodName``; it is False for a
    method-name-only pattern (e.g. ``*get*Index``), matched against the bare
    method name.  Only the last one or two dot-separated segments of the
    declaring-type+method pattern are used — see module docstring for why
    the package portion is dropped.  Returns ``None`` when the expression
    cannot be safely reduced (e.g. no parameter-pattern parens found), or
    when discarding the package portion left a pattern that matches
    virtually every method (see ``_UNIVERSAL_REGEXES`` below).
    """
    m = _EXECUTION_POINTCUT.match(expr)
    if not m:
        return None
    inner = m.group(1).strip()

    # inner = "<ret> <declaring-type?.method-name-pattern>(<params>)"
    m2 = re.match(r"^(.*)\((.*)\)\s*$", inner, re.DOTALL)
    if not m2:
        return None
    before_params = m2.group(1).strip()
    tokens = before_params.split()
    if not tokens:
        return None
    method_class_pattern = tokens[-1]

    segments = method_class_pattern.split(".")
    if len(segments) == 1:
        target_pattern = segments[0]
        needs_class = False
    else:
        target_pattern = f"{segments[-2]}.{segments[-1]}"
        needs_class = True

    if not target_pattern:
        return None
    regex_str = "^" + _aspectj_pattern_to_regex(target_pattern) + "$"
    if regex_str in _UNIVERSAL_REGEXES:
        return None
    return regex_str, needs_class


def _resolve_pointcut_expr(
    expr: str, pointcuts: dict[str, str],
) -> Optional[str]:
    """Follow a same-class named ``@Pointcut`` reference to its expression.

    Returns the expression to actually parse (``expr`` itself when it is
    already an inline expression), or ``None`` when ``expr`` is a
    cross-aspect reference, an unresolvable reference, or a compound boolean
    expression — all out of scope for this pass. The compound check is
    applied both to ``expr`` and, when ``expr`` is a named reference, to the
    ``@Pointcut`` body it resolves to — a bare reference like
    ``"myPointcut()"`` carries no ``&&``/``||`` itself even when the
    pointcut it names does.
    """
    if any(op in expr for op in _COMPOUND_OPERATORS):
        return None

    ref = _NAMED_POINTCUT_REF.match(expr)
    if ref:
        name = ref.group(1)
        resolved = pointcuts.get(name)
        if resolved is None or any(op in resolved for op in _COMPOUND_OPERATORS):
            return None
        return resolved

    if "." in expr and "(" in expr and not expr.lstrip().startswith(
        ("execution(", "@annotation(", "within(", "args(", "target(", "this("),
    ):
        # Looks like a dotted "Other.pointcut()" cross-aspect reference.
        return None

    return expr


def _load_extra(raw: Optional[str]) -> dict:
    try:
        return json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def resolve_aop_advice(store: "GraphStore") -> dict:
    """Resolve Spring AOP advice methods to the target methods their pointcut matches.

    Safe to call multiple times: previously derived edges (tagged
    ``extra.aop_resolved``) are cleared and rebuilt on every call, so a
    changed or removed pointcut cannot leave a stale edge behind.

    Returns a dict with resolution counts for telemetry.
    """
    conn = store._conn

    java_files: set[str] = {
        row["file_path"]
        for row in conn.execute(
            "SELECT DISTINCT file_path FROM nodes WHERE language = 'java'"
        ).fetchall()
    }
    if not java_files:
        return {
            "aspects_indexed": 0,
            "advice_resolved": 0,
            "calls_created": 0,
            "stale_calls_removed": 0,
        }

    # -------------------------------------------------------------------
    # Clear previously derived edges so stale advice→target links from a
    # since-changed pointcut don't linger.
    # -------------------------------------------------------------------
    stale_ids: list[tuple[int]] = []
    for row in conn.execute(
        "SELECT id, extra FROM edges WHERE kind = 'CALLS'"
    ).fetchall():
        if _load_extra(row["extra"]).get(_DERIVED_FLAG):
            stale_ids.append((row["id"],))
    if stale_ids:
        conn.executemany("DELETE FROM edges WHERE id = ?", stale_ids)

    # -------------------------------------------------------------------
    # Index @Aspect classes and every Java method, grouped by (class, file).
    # -------------------------------------------------------------------
    aspect_classes: list[dict] = []
    for row in conn.execute(
        "SELECT name, qualified_name, file_path, extra FROM nodes "
        "WHERE kind = 'Class' AND language = 'java'"
    ).fetchall():
        extra = _load_extra(row["extra"])
        if "Aspect" in (extra.get("spring_annotations") or []):
            aspect_classes.append({
                "name": row["name"],
                "qualified_name": row["qualified_name"],
                "file_path": row["file_path"],
            })

    if not aspect_classes:
        logger.info("AOP resolver: no @Aspect classes found, skipping")
        return {
            "aspects_indexed": 0,
            "advice_resolved": 0,
            "calls_created": 0,
            "stale_calls_removed": len(stale_ids),
        }

    all_methods: list[dict] = []
    methods_by_class_file: dict[tuple[str, str], list[dict]] = {}
    for row in conn.execute(
        "SELECT name, qualified_name, parent_name, file_path, extra FROM nodes "
        "WHERE kind IN ('Function', 'Test') AND language = 'java' "
        "AND parent_name IS NOT NULL"
    ).fetchall():
        extra = _load_extra(row["extra"])
        entry = {
            "name": row["name"],
            "qualified_name": row["qualified_name"],
            "parent_name": row["parent_name"],
            "file_path": row["file_path"],
            "decorators": extra.get("decorators") or [],
        }
        all_methods.append(entry)
        methods_by_class_file.setdefault(
            (row["parent_name"], row["file_path"]), [],
        ).append(entry)

    advice_resolved = 0
    calls_created = 0

    for aspect in aspect_classes:
        methods = methods_by_class_file.get(
            (aspect["name"], aspect["file_path"]), [],
        )

        pointcuts: dict[str, str] = {}
        advices: list[tuple[dict, str, str]] = []
        for method in methods:
            for deco in method["decorators"]:
                head = _annotation_head(deco)
                if head == "Pointcut":
                    pc_expr = _extract_annotation_string_arg(deco)
                    if pc_expr:
                        pointcuts[method["name"]] = pc_expr
                elif head in _ADVICE_ANNOTATIONS:
                    adv_expr = _extract_annotation_string_arg(deco)
                    if adv_expr:
                        advices.append((method, head, adv_expr))

        for advice_method, advice_kind, raw_expr in advices:
            expr = _resolve_pointcut_expr(raw_expr, pointcuts)
            if not expr:
                continue

            ann_match = _ANNOTATION_POINTCUT.match(expr)
            if ann_match:
                fqcn = ann_match.group(1).strip()
                bare_name = fqcn.rsplit(".", 1)[-1].strip()
                if not bare_name:
                    continue
                pointcut_kind = "@annotation"
                # Method-level match only — a class-level stereotype
                # annotation (e.g. @Transactional) is @within() semantics,
                # a different pointcut designator, and is not expanded here
                # (see module docstring).
                targets = [
                    candidate
                    for candidate in all_methods
                    if any(
                        _annotation_head(d) == bare_name
                        for d in candidate["decorators"]
                    )
                ]
            else:
                parsed = _parse_execution_expression(expr)
                if not parsed:
                    continue
                regex_str, needs_class = parsed
                pointcut_kind = "execution"
                try:
                    pattern = re.compile(regex_str)
                except re.error:
                    continue
                targets = []
                for candidate in all_methods:
                    if needs_class:
                        if not candidate["parent_name"]:
                            continue
                        subject = f"{candidate['parent_name']}.{candidate['name']}"
                    else:
                        subject = candidate["name"]
                    if pattern.match(subject):
                        targets.append(candidate)

            if not targets:
                continue

            advice_resolved += 1
            for target in targets:
                if target["qualified_name"] == advice_method["qualified_name"]:
                    continue
                store.upsert_edge(EdgeInfo(
                    kind="CALLS",
                    source=advice_method["qualified_name"],
                    target=target["qualified_name"],
                    file_path=advice_method["file_path"],
                    line=0,
                    extra={
                        _DERIVED_FLAG: True,
                        "pointcut_kind": pointcut_kind,
                        "pointcut_expr": expr,
                        "advice_kind": advice_kind,
                        "confidence": 0.7,
                        "confidence_tier": "INFERRED",
                    },
                ))
                calls_created += 1

    store.commit()
    logger.info(
        "AOP resolver: %d @Aspect classes, %d advice resolved, %d CALLS edges created",
        len(aspect_classes), advice_resolved, calls_created,
    )
    return {
        "aspects_indexed": len(aspect_classes),
        "advice_resolved": advice_resolved,
        "calls_created": calls_created,
        "stale_calls_removed": len(stale_ids),
    }
