from neurograph.indexer.sb_facts import DOC_EXACT, DOC_INFERRED, extract_sb_facts


def test_extracts_korean_sb_api_validation_status_and_errors():
    text = "\n".join(
        [
            "결제 화면",
            "사용자는 amount_cents 필드에 금액을 입력한다.",
            "POST /api/payments 호출 후 status 값은 PAID, FAILED, PENDING 중 하나다.",
            "비밀번호는 8자 이상이어야 하고, 이메일 형식 검증은 required.",
            "실패 시 오류 메시지: \"결제 실패\"",
            "관리자만 APPROVED 상태로 환불을 승인할 수 있다.",
            "payments 테이블에 저장한다.",
            "Stripe 연동을 사용한다.",
        ]
    )

    facts = extract_sb_facts(text, page=3)
    by_type = _facts_by_type(facts)

    assert _values(by_type["api_reference"]) == {"/api/payments", "POST"}
    assert {"PAID", "FAILED", "PENDING", "APPROVED"}.issubset(_values(by_type["status_value"]))
    assert {"amount_cents", "status"}.issubset(_values(by_type["form_field"]))
    assert any(fact.type == "validation_rule" and "8자 이상" in fact.value for fact in facts)
    assert any(fact.type == "validation_rule" and "이메일 형식" in fact.evidence_quote for fact in facts)
    assert any(fact.type == "error_message" and fact.value == "결제 실패" and fact.confidence == DOC_EXACT for fact in facts)
    assert any(fact.type == "permission_rule" and "관리자만" in fact.evidence_quote for fact in facts)
    assert any(fact.type == "database_entity" and fact.value == "payments" for fact in facts)
    assert any(fact.type == "external_integration" and fact.value == "Stripe" for fact in facts)
    assert all(fact.page == 3 and fact.evidence_quote and fact.confidence in {DOC_EXACT, DOC_INFERRED, "AMBIGUOUS"} for fact in facts)


def test_extracts_markdown_section_line_evidence_without_code_links():
    text = "GET /api/orders 는 주문 목록을 반환한다.\nstatus 필드는 APPROVED 또는 FAILED 값만 허용한다.\n최대 3회 재시도한다."

    facts = extract_sb_facts(text, start_line=10, end_line=12)

    assert any(fact.type == "api_reference" and fact.subject == "/api/orders" for fact in facts)
    assert any(fact.type == "form_field" and fact.value == "status" for fact in facts)
    assert any(fact.type == "validation_rule" and "최대 3회" in fact.value for fact in facts)
    assert all(fact.start_line is not None and 10 <= fact.start_line <= 12 for fact in facts)
    assert all("code_path" not in fact.metadata and "symbol_id" not in fact.metadata for fact in facts)


def test_does_not_over_extract_vague_business_text():
    facts = extract_sb_facts("고객 만족도를 높이고 더 편리한 경험을 제공한다. 향후 화면을 개선한다.", page=1)

    assert facts == []


def _facts_by_type(facts):
    by_type = {}
    for fact in facts:
        by_type.setdefault(fact.type, []).append(fact)
    return by_type


def _values(facts):
    return {fact.value for fact in facts}
