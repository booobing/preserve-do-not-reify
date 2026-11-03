"""Sentence / query templates shared by the synthetic world generator and the template extractor.

Kept separate so `extract.py` does not depend on `synthetic.py` (design §3). v1 text extraction is
template-closed by construction: the extractor inverts exactly these templates.
"""

from __future__ import annotations

# Sentence templates are reverse-parsable by extract.TemplateTextExtractor: every slot is an entity
# name matched by NAME_PATTERN and is delimited by whitespace/punctuation/Korean particles.
NAME_PATTERN = r"[A-Za-z0-9\-]+"

SENTENCE_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "DefectEvent": {
        "en": [
            "At {zone}, {equipment} raised a defect: {sensor_primary} and {sensor_secondary} both crossed their thresholds, so the prescribed action was {action}.",
            "A defect event on {equipment} in {zone} involved sensors {sensor_primary} and {sensor_secondary}; the response was {action}.",
            "{equipment} ({zone}) reported {sensor_primary} together with {sensor_secondary} out of range, and {action} was applied.",
        ],
        "ko": [
            "{zone} 구역의 {equipment}에서 {sensor_primary}와 {sensor_secondary}가 동시에 임계값을 넘어 {action} 조치가 지정되었다.",
            "{equipment}의 결함 이벤트는 {zone}에서 발생했고 센서 {sensor_primary}, {sensor_secondary}가 관여했으며 대응은 {action}이었다.",
        ],
    },
    "Maintenance": {
        "en": [
            "Technician {technician} serviced {equipment} in {zone} after error code {error_code}.",
            "{technician} performed maintenance on {equipment} at {zone}, closing {error_code}.",
            "Maintenance record: {equipment} in {zone} was handled by {technician} for {error_code}.",
        ],
        "ko": [
            "기술자 {technician}가 {zone}의 {equipment}를 오류 코드 {error_code} 때문에 정비했다.",
            "{zone}에 있는 {equipment}는 {error_code} 이후 {technician}가 유지보수를 수행했다.",
        ],
    },
    "Escalation": {
        "en": [
            "Error code {error_code} was escalated to {technician}, who ordered {action}.",
            "{technician} received the escalation for {error_code} and chose {action}.",
            "Escalation: {error_code} -> {technician} -> {action}.",
        ],
        "ko": [
            "오류 코드 {error_code}는 {technician}에게 에스컬레이션되었고 {action} 조치가 내려졌다.",
            "{technician}가 {error_code} 에스컬레이션을 받아 {action}을 선택했다.",
        ],
    },
}

# Retrieval query templates deliberately use different phrasing from the sentences and mention a
# subset of roles; the generator only keeps a query if that subset identifies the fact uniquely.
RETRIEVAL_QUERY_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "DefectEvent": {
        "en": [
            "Which action was assigned for the {equipment} defect in {zone} involving {sensor_primary} and {sensor_secondary}?",
            "What happened when {sensor_primary} and {sensor_secondary} tripped on {equipment}?",
            "Find the defect event in {zone} on {equipment} that ended with {action}.",
        ],
        "ko": [
            "{zone}에서 {equipment}에 {sensor_primary}, {sensor_secondary} 이상이 났을 때 어떤 조치가 배정됐는가?",
            "{equipment}에서 {sensor_primary}와 {sensor_secondary}가 함께 튀었을 때 무슨 일이 있었나?",
        ],
    },
    "Maintenance": {
        "en": [
            "Who serviced {equipment} in {zone} for {error_code}?",
            "Which error code did {technician} close on {equipment}?",
            "Where did {technician} handle {error_code}?",
        ],
        "ko": [
            "{zone}의 {equipment}를 {error_code} 때문에 누가 정비했나?",
            "{technician}가 {equipment}에서 닫은 오류 코드는 무엇인가?",
        ],
    },
    "Escalation": {
        "en": [
            "What did {technician} order after the escalation of {error_code}?",
            "Who handled the escalation of {error_code} and with which action?",
            "Which escalation led {technician} to choose {action}?",
        ],
        "ko": [
            "{error_code} 에스컬레이션 뒤 {technician}는 무엇을 지시했나?",
            "{technician}가 {action}을 고른 에스컬레이션은 어느 것인가?",
        ],
    },
}

COMPLETION_QUESTION_TEMPLATES: dict[str, str] = {
    "en": "In the {relation} where {context}, what is the {masked_role}?",
    "ko": "{context} 인 {relation}에서 {masked_role}는 무엇인가?",
}
