import re
from pathlib import Path

from code_review_graph.aop_resolver import (
    _aspectj_pattern_to_regex,
    _parse_execution_expression,
    resolve_aop_advice,
)
from code_review_graph.graph import GraphStore
from code_review_graph.parser import CodeParser, EdgeInfo
from code_review_graph.tools.query import query_graph

ANNOTATION_SOURCE = """
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;

@Aspect
class AccessLimitAspect {
    @Before("@annotation(com.foo.common.annotation.AccessLimit)")
    void beforeLimit() {}
}

class OrderController {
    @AccessLimit
    void placeOrder() {}

    void unrelatedMethod() {}
}
"""

EXECUTION_SOURCE = """
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Around;

@Aspect
class LoggingAspect {
    @Around("execution(* com.foo.service.PaymentService.*(..))")
    Object logAround() { return null; }
}

class PaymentService {
    void pay() {}
}

class OtherService {
    void unrelated() {}
}
"""

NAMED_POINTCUT_SOURCE = """
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Pointcut;
import org.aspectj.lang.annotation.Before;
import org.aspectj.lang.annotation.AfterThrowing;

@Aspect
class AccessLimitAspect {
    @Pointcut("@annotation(com.foo.common.annotation.AccessLimit)")
    void accessLimitPointcut() {}

    @Before("accessLimitPointcut()")
    void before(JoinPoint jp) {}

    @AfterThrowing(pointcut = "accessLimitPointcut()", throwing = "ex")
    void afterThrowing(JoinPoint jp, Throwable ex) {}
}

class OrderController {
    @AccessLimit
    void placeOrder() {}
}
"""

COMPOUND_SOURCE = """
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;

@Aspect
class SecurityAspect {
    @Before("within(com.foo.service..*) && !execution(* com.foo.service.internal.*.*(..))")
    void checkSecurity() {}
}

class SomeService {
    void doWork() {}
}
"""


def _build_store(tmp_path: Path, source: str, filename: str = "Aspect.java"):
    path = tmp_path / filename
    nodes, edges = CodeParser().parse_bytes(path, source.encode())
    graph_dir = tmp_path / ".code-review-graph"
    graph_dir.mkdir(exist_ok=True)
    db_path = graph_dir / "graph.db"
    store = GraphStore(db_path)
    store.store_file_nodes_edges(str(path), nodes, edges, "hash")
    return path, store


def test_annotation_pointcut_before_advice_creates_calls_edge(tmp_path: Path) -> None:
    path, store = _build_store(tmp_path, ANNOTATION_SOURCE)
    try:
        stats = resolve_aop_advice(store)
        assert stats["aspects_indexed"] == 1
        assert stats["calls_created"] == 1

        source_qual = f"{path.as_posix()}::AccessLimitAspect.beforeLimit"
        edges = store.get_edges_by_source(source_qual)
        calls = [e for e in edges if e.kind == "CALLS"]
        assert len(calls) == 1
        edge = calls[0]
        assert edge.target_qualified == f"{path.as_posix()}::OrderController.placeOrder"
        assert edge.extra["aop_resolved"] is True
        assert edge.extra["pointcut_kind"] == "@annotation"
        assert edge.extra["pointcut_expr"] == (
            "@annotation(com.foo.common.annotation.AccessLimit)"
        )
    finally:
        store.close()


def test_execution_pointcut_around_advice_creates_calls_edge(tmp_path: Path) -> None:
    path, store = _build_store(tmp_path, EXECUTION_SOURCE)
    try:
        stats = resolve_aop_advice(store)
        assert stats["calls_created"] == 1

        source_qual = f"{path.as_posix()}::LoggingAspect.logAround"
        edges = [e for e in store.get_edges_by_source(source_qual) if e.kind == "CALLS"]
        assert len(edges) == 1
        edge = edges[0]
        assert edge.target_qualified == f"{path.as_posix()}::PaymentService.pay"
        assert edge.extra["pointcut_kind"] == "execution"
        assert edge.extra["aop_resolved"] is True
    finally:
        store.close()


def test_named_pointcut_reference_resolves_both_advices(tmp_path: Path) -> None:
    path, store = _build_store(tmp_path, NAMED_POINTCUT_SOURCE)
    try:
        stats = resolve_aop_advice(store)
        assert stats["calls_created"] == 2

        target_qual = f"{path.as_posix()}::OrderController.placeOrder"

        before_qual = f"{path.as_posix()}::AccessLimitAspect.before"
        before_edges = [
            e for e in store.get_edges_by_source(before_qual) if e.kind == "CALLS"
        ]
        assert len(before_edges) == 1
        assert before_edges[0].target_qualified == target_qual
        assert before_edges[0].extra["pointcut_expr"] == (
            "@annotation(com.foo.common.annotation.AccessLimit)"
        )

        after_throwing_qual = f"{path.as_posix()}::AccessLimitAspect.afterThrowing"
        after_edges = [
            e for e in store.get_edges_by_source(after_throwing_qual) if e.kind == "CALLS"
        ]
        assert len(after_edges) == 1
        assert after_edges[0].target_qualified == target_qual
    finally:
        store.close()


def test_compound_boolean_pointcut_is_safely_skipped(tmp_path: Path) -> None:
    path, store = _build_store(tmp_path, COMPOUND_SOURCE)
    try:
        stats = resolve_aop_advice(store)
        assert stats["calls_created"] == 0
        assert stats["advice_resolved"] == 0

        source_qual = f"{path.as_posix()}::SecurityAspect.checkSecurity"
        edges = [e for e in store.get_edges_by_source(source_qual) if e.kind == "CALLS"]
        assert edges == []
    finally:
        store.close()


def test_callers_of_query_finds_advice_method(tmp_path: Path) -> None:
    path, store = _build_store(tmp_path, ANNOTATION_SOURCE)
    try:
        resolve_aop_advice(store)
    finally:
        store.close()

    target_qual = f"{path.as_posix()}::OrderController.placeOrder"
    result = query_graph("callers_of", target_qual, repo_root=str(tmp_path))
    assert result["status"] == "ok"
    assert result["result_count"] == 1
    assert result["results"][0]["name"] == "beforeLimit"
    assert {edge["kind"] for edge in result["edges"]} == {"CALLS"}


def test_advises_and_advised_by_filter_out_plain_calls_edges(tmp_path: Path) -> None:
    """``advises``/``advised_by`` must return only aop_resolved edges, while
    ``callers_of`` keeps returning every CALLS edge regardless of provenance.
    """
    path, store = _build_store(tmp_path, ANNOTATION_SOURCE)
    try:
        resolve_aop_advice(store)

        target_qual = f"{path.as_posix()}::OrderController.placeOrder"
        advice_qual = f"{path.as_posix()}::AccessLimitAspect.beforeLimit"
        # A regular, non-AOP caller of the same target method.
        plain_caller_qual = f"{path.as_posix()}::OrderController.unrelatedMethod"
        store.upsert_edge(EdgeInfo(
            kind="CALLS",
            source=plain_caller_qual,
            target=target_qual,
            file_path=str(path),
            line=1,
        ))
        store.commit()
    finally:
        store.close()

    callers = query_graph("callers_of", target_qual, repo_root=str(tmp_path))
    assert callers["result_count"] == 2
    assert {r["name"] for r in callers["results"]} == {"beforeLimit", "unrelatedMethod"}

    advised_by = query_graph("advised_by", target_qual, repo_root=str(tmp_path))
    assert advised_by["status"] == "ok"
    assert advised_by["result_count"] == 1
    assert advised_by["results"][0]["name"] == "beforeLimit"
    assert advised_by["edges"][0]["kind"] == "CALLS"

    advises = query_graph("advises", advice_qual, repo_root=str(tmp_path))
    assert advises["status"] == "ok"
    assert advises["result_count"] == 1
    assert advises["results"][0]["name"] == "placeOrder"

    # The plain caller has no AOP relationships at all in either direction.
    assert query_graph("advises", plain_caller_qual, repo_root=str(tmp_path))["result_count"] == 0


class TestAspectjPatternToRegex:
    """Direct unit tests for the three documented conversion rules."""

    def test_literal_pattern_matches_only_itself(self) -> None:
        pattern = re.compile("^" + _aspectj_pattern_to_regex("PaymentService") + "$")
        assert pattern.match("PaymentService")
        assert not pattern.match("PaymentServiceImpl")
        assert not pattern.match("paymentservice")

    def test_single_star_does_not_cross_a_dot_boundary(self) -> None:
        pattern = re.compile("^" + _aspectj_pattern_to_regex("*Service") + "$")
        assert pattern.match("PaymentService")
        assert pattern.match("Service")
        assert not pattern.match("internal.Service")

    def test_double_dot_becomes_a_permissive_any_match(self) -> None:
        # ".." is documented as "any number of intervening path segments", but
        # it compiles to a bare ".*" — it matches any characters at all, not
        # only well-formed dot-separated segments. This test pins the actual
        # (looser) behavior so a future change to the translation is a
        # deliberate, visible diff rather than a silent regression either way.
        pattern = re.compile("^" + _aspectj_pattern_to_regex("com..Service") + "$")
        assert pattern.match("com.foo.bar.Service")
        assert pattern.match("comXService")  # crosses non-dot characters too

    def test_special_regex_characters_are_escaped(self) -> None:
        pattern = re.compile("^" + _aspectj_pattern_to_regex("Foo$Bar") + "$")
        assert pattern.match("Foo$Bar")
        assert not pattern.match("FooXBar")


class TestParseExecutionExpression:
    def test_package_and_class_wildcard_needs_class_true(self) -> None:
        result = _parse_execution_expression(
            "execution(* com.foo.service.PaymentService.*(..))"
        )
        assert result is not None
        regex_str, needs_class = result
        assert needs_class is True
        pattern = re.compile(regex_str)
        assert pattern.match("PaymentService.pay")
        assert not pattern.match("OtherService.pay")

    def test_method_name_only_pattern_needs_class_false(self) -> None:
        result = _parse_execution_expression("execution(* *get*Index(..))")
        assert result is not None
        regex_str, needs_class = result
        assert needs_class is False
        pattern = re.compile(regex_str)
        assert pattern.match("getUserIndex")
        assert not pattern.match("createUser")

    def test_non_execution_expression_returns_none(self) -> None:
        assert _parse_execution_expression("@annotation(com.foo.Bar)") is None

    def test_missing_parameter_parens_returns_none(self) -> None:
        assert _parse_execution_expression("execution(* com.foo.Bar.baz)") is None
