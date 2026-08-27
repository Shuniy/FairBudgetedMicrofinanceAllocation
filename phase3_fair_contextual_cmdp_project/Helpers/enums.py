from __future__ import annotations

from enum import Enum


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class IncomeGroup(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    LOWER_MIDDLE = "lower_middle"
    MIDDLE = "middle"
    UPPER_MIDDLE = "upper_middle"
    HIGH = "high"


class ProtectedGroup(str, Enum):
    WOMEN = "women"
    DISABLED = "disabled"
    AGED = "aged"
    MINORITY = "minority"
    RESERVED = "reserved"
    NONE = "none"


class RegionType(str, Enum):
    URBAN = "urban"
    SEMI_URBAN = "semi_urban"
    RURAL = "rural"
    ISOLATED = "isolated"


class LoanPurpose(str, Enum):
    BUSINESS = "business"
    AGRICULTURE = "agriculture"
    EDUCATION = "education"
    HEALTH = "health"
    HOUSING = "housing"
    OTHER = "other"


class Sector(str, Enum):
    AGRICULTURE = "agriculture"
    RETAIL = "retail"
    SERVICES = "services"
    MANUFACTURING = "manufacturing"
    OTHER = "other"
