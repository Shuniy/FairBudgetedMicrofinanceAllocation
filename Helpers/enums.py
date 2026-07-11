from enum import Enum

"""
Application Enums
"""


class SectorType(Enum):
    AGRICULTURE = "agriculture"
    MANUFACTURING = "manufacturing"
    SERVICES = "services"
    RETAIL = "retail"
    PERSONAL = "personal"
    OTHER = "other"


class LoanPurpose(Enum):
    STARTUP = "startup"
    WORKING_CAPITAL = "working_capital"
    BUSINESS_EXPANSION = "business_expansion"
    EQUIPMENT_PURCHASE = "equipment_purchase"
    REFINANCING = "refinancing"
    EDUCATION = "education"
    HEALTHCARE = "healthcare"
    EMERGENCY = "emergency"
    HOUSING = "housing"
    OTHER = "other"


class RegionType(Enum):
    URBAN = "urban"
    RURAL = "rural"
    SEMI_URBAN = "semi_urban"
    ISOLATED = "isolated"


class IncomeGroup(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    LOWER_MIDDLE = "lower_middle"
    MIDDLE = "middle"
    UPPER_MIDDLE = "upper_middle"
    HIGH = "high"


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class ProtectedGroup(Enum):
    RESERVED = "reserved"  # such as tribal people etc
    MINORITY = "minority"  # such as religious minorities, transgender, immigrants etc
    WOMEN = "women"  # women
    DISABLED = "disabled"  # disabled people
    AGED = "aged"  # aged people
    NONE = "none"  # not belonging to any protected group


class Region(Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    CENTRAL = "central"
