"""Structured output schemas for the LLM steps."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

TodBucket = Literal[
    "early_morning",
    "late_morning",
    "afternoon",
    "evening",
    "night_before_midnight",
    "night_after_midnight",
]
DowBucket = Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
AnalysisType = Literal["trend", "compare_periods", "timing", "time_of_day_profile", "day_of_week_profile", "overall_summary"]
TimingEvent = Literal["first", "last", "all"]
Granularity = Literal["day", "week", "month"]


class Concept(BaseModel):
    id: str = Field(description="snake_case identifier, e.g. 'sedentary_living_room'")
    name: str = Field(description="Short human-readable behavior name")
    physical_definition: str = Field(
        description="One sentence describing what this behavior looks like in THIS home's sensors. "
        "Shown verbatim to human raters."
    )
    detectable: bool
    detectability_note: str = Field(description="Why the behavior is or is not observable with the available sensors")
    retrieval_queries: List[str] = Field(
        description="1-6 caption-style sentences describing observable sensor behavior. No sensor IDs, no activity labels."
    )
    time_of_day: List[TodBucket] = Field(
        default_factory=list,
        description="Hard time-of-day filter applied after retrieval. Empty = no filter.",
    )
    days_of_week: List[DowBucket] = Field(
        default_factory=list,
        description="Hard day-of-week filter applied after retrieval. Empty = no filter.",
    )
    rooms: List[str] = Field(
        default_factory=list,
        description="Hard filter: window must visit at least one of these rooms (exact room ids from the "
        "inventory). Use only when the behavior is inherently tied to specific rooms. Empty = no filter.",
    )
    required_sensors: List[str] = Field(
        default_factory=list,
        description="Hard filter: window must contain an activation of at least one of these sensor ids. "
        "Use only when the behavior is defined by a specific fixture event (e.g. an entrance door). Empty = no filter.",
    )
    timing_hours: List[int] = Field(
        default_factory=list,
        description="Only for timing analyses: [start_hour, end_hour) window in which an occurrence must START "
        "(e.g. breakfast [5, 11], lunch [11, 15], dinner [16, 21]; may wrap midnight, e.g. bedtime [20, 3]). "
        "Sessions are built from matches at any time, so activity crossing a window edge is not cut. "
        "Empty = derived from time_of_day.",
    )


class Period(BaseModel):
    label: str
    start_date: str = Field(description="YYYY-MM-DD, inclusive")
    end_date: str = Field(description="YYYY-MM-DD, inclusive")


class AnalysisSpec(BaseModel):
    type: AnalysisType
    granularity: Granularity = "week"
    periods: List[Period] = Field(
        default_factory=list,
        description="For compare_periods and timing: 2+ periods within the data range. Empty = first vs last third.",
    )
    timing_event: TimingEvent = Field(
        default="first",
        description="Only for timing: which occurrence per day is timed. 'first' (e.g. wake-up), 'last' "
        "(e.g. bedtime), 'all' (every episode, e.g. meals or medication: typical times and consistency).",
    )
    day_boundary_hour: int = Field(
        default=4, ge=0, le=23,
        description="Only for timing: clock hour at which a day starts. Default 4 = activity between midnight "
        "and 04:00 counts toward the previous day (so a 01:00 bedtime is that evening's bedtime).",
    )
    session_gap_minutes: float = Field(
        default=2, ge=0, le=120,
        description="Only for timing: matched activity separated by at most this many minutes is one session "
        "(e.g. 30 for a meal: cooking, eating and clearing up). Default 2 = raw episodes.",
    )
    min_session_minutes: float = Field(
        default=0, ge=0, le=120,
        description="Only for timing: ignore sessions shorter than this (e.g. 5 for a meal, so passing through "
        "the kitchen does not count).",
    )


class AnalysisPlan(BaseModel):
    interpretation: str = Field(description="What the user is asking, restated in one or two sentences")
    concepts: List[Concept]
    scope_start_date: Optional[str] = Field(default=None, description="YYYY-MM-DD or null for start of data")
    scope_end_date: Optional[str] = Field(default=None, description="YYYY-MM-DD or null for end of data")
    analyses: List[AnalysisSpec] = Field(description="Primary analysis first")
    caveats: List[str] = Field(default_factory=list)


FindingKind = Literal["level", "change", "timing", "comparison", "pattern"]


class Finding(BaseModel):
    concept_id: str = Field(description="id of the analyzed concept this finding is about, copied exactly")
    kind: FindingKind = Field(
        description="level = typical amount/frequency; change = trend or period difference (or its absence); "
        "timing = typical clock time / spread / days observed; comparison = between concepts; "
        "pattern = time-of-day or day-of-week distribution"
    )
    statement: str = Field(
        description="One atomic, checkable claim in plain language (one number or one direction), "
        "e.g. 'About 27 minutes per day of in-bed presence' or 'In-bed presence did not change over the 7 weeks'"
    )


class Report(BaseModel):
    headline: str = Field(description="One-sentence direct answer to the question")
    body: str = Field(description="Short report, 3-6 sentences, plain prose")
    findings: List[Finding] = Field(
        default_factory=list,
        description="2-8 atomic findings covering every claim of the headline and body for analyzed concepts",
    )
    limitations: List[str] = Field(default_factory=list, description="0-3 short limitations grounded in the provided data")


class ShortAnswer(BaseModel):
    key: str = Field(description="The item key, copied exactly")
    answer: str = Field(description="1-3 word answer, faithful to the headline")


class ShortAnswers(BaseModel):
    answers: List[ShortAnswer]
