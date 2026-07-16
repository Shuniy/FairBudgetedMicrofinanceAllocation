from __future__ import annotations

from enum import Enum


class Gender(str, Enum):
    """gender options."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class IncomeGroup(str, Enum):
    """income group options."""
    VERY_LOW = "very_low"
    LOW = "low"
    LOWER_MIDDLE = "lower_middle"
    MIDDLE = "middle"
    UPPER_MIDDLE = "upper_middle"
    HIGH = "high"


class ProtectedGroup(str, Enum):
    """protected group options."""
    WOMEN = "women"
    DISABLED = "disabled"
    AGED = "aged"
    MINORITY = "minority"
    RESERVED = "reserved"
    NONE = "none"


class RegionType(str, Enum):
    """region type options."""
    URBAN = "urban"
    SEMI_URBAN = "semi_urban"
    RURAL = "rural"
    ISOLATED = "isolated"


class Region(str, Enum):
    """region options."""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    CENTRAL = "central"


class LoanPurpose(str, Enum):
    """loan purpose options."""
    BUSINESS = "business"
    AGRICULTURE = "agriculture"
    EDUCATION = "education"
    HEALTH = "health"
    HOUSING = "housing"
    OTHER = "other"


class Sector(str, Enum):
    """sector options."""
    AGRICULTURE = "agriculture"
    RETAIL = "retail"
    SERVICES = "services"
    MANUFACTURING = "manufacturing"
    OTHER = "other"


# Backward-compatible name used by the original parent models.
SectorType = Sector
