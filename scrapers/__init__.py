from .base import BaseScraper
from .indeed import IndeedScraper
from .linkedin import LinkedInScraper
from .glassdoor import GlassdoorScraper
from .stepstone import StepstoneScraper
from .remotive import RemotiveScraper
from .adzuna import AdzunaScraper
from .jsearch import JSearchScraper
from .linkedin_guest import LinkedInGuestScraper
from .arbeitnow import ArbeitnowScraper
from .themuse import TheMuseScraper
from .himalayas import HimalayasScraper
from .jobspy_wrapper import JobSpyIndeedScraper, JobSpyGlassdoorScraper, JobSpyGoogleScraper
from .greenhouse import GreenhouseScraper
from .lever import LeverScraper
from .linkedin_posts import LinkedInPostsScraper
from .internet_search import InternetSearchScraper

SCRAPERS = {
    # JobSpy-backed scrapers (handle JS rendering + bot detection)
    "indeed": JobSpyIndeedScraper,
    "glassdoor": JobSpyGlassdoorScraper,
    "google": JobSpyGoogleScraper,
    # Company ATS scrapers (free public APIs)
    "greenhouse": GreenhouseScraper,
    "lever": LeverScraper,
    "linkedin_posts": LinkedInPostsScraper,
    "internet": InternetSearchScraper,
    # Other boards
    "linkedin": LinkedInGuestScraper,
    "stepstone": StepstoneScraper,
    "remotive": RemotiveScraper,
    "adzuna": AdzunaScraper,
    "jsearch": JSearchScraper,
    "arbeitnow": ArbeitnowScraper,
    "themuse": TheMuseScraper,
    "himalayas": HimalayasScraper,
}
