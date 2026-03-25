"""Data models for the job application pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List


class JobBoard(Enum):
    INDEED = "indeed"
    LINKEDIN = "linkedin"
    GLASSDOOR = "glassdoor"
    STEPSTONE = "stepstone"
    REMOTIVE = "remotive"
    ADZUNA = "adzuna"
    JSEARCH = "jsearch"
    ARBEITNOW = "arbeitnow"
    THEMUSE = "themuse"
    HIMALAYAS = "himalayas"
    GOOGLE = "google"
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    LINKEDIN_POSTS = "linkedin_posts"
    INTERNET = "internet"


@dataclass
class Job:
    title: str
    company: str
    location: str
    url: str
    board: JobBoard
    description: str = ""
    salary: str = ""
    date_posted: str = ""
    job_type: str = ""  # full-time, part-time, contract
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())
    match_score: float = 0.0
    match_details: Dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Unique identifier based on URL."""
        return self.url

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "url": self.url,
            "board": self.board.value,
            "description": self.description,
            "salary": self.salary,
            "date_posted": self.date_posted,
            "job_type": self.job_type,
            "scraped_at": self.scraped_at,
            "match_score": self.match_score,
            "match_details": self.match_details,
        }


@dataclass
class SearchQuery:
    keywords: str
    location: str = ""
    remote: bool = False
    job_type: str = ""  # full-time, part-time, contract
    max_age_days: int = 30
    boards: List[JobBoard] = field(default_factory=lambda: list(JobBoard))
