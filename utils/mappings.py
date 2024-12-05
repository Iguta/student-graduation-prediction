# utils/course_mappings.py
# Application mode mapping
APPLICATION_MODE_MAPPING = {
    1: '1st phase - general',
    2: 'Ordinance 612/93',
    5: '1st phase - Azores',
    7: 'Other higher courses',
    10: 'Ordinance 854-B/99',
    15: 'Intl. student (bachelor)',
    16: '1st phase - Madeira',
    17: '2nd phase - general',
    18: '3rd phase - general',
    26: 'Ordinance 533-A/99 (Plan)',
    27: 'Ordinance 533-A/99 (Institution)',
    39: 'Over 23 years old',
    42: 'Transfer',
    43: 'Change of course',
    44: 'Technological diploma',
    51: 'Change institution/course',
    53: 'Short cycle diploma',
    57: 'Change institution (Intl.)'
}

# Course mappings
COURSE_MAPPING = {
    33: 'Biofuel Production Tech',
    171: 'Animation & Multimedia',
    8014: 'Social Service (evening)',
    9003: 'Agronomy',
    9070: 'Comm. Design',
    9085: 'Veterinary Nursing',
    9119: 'Informatics Eng.',
    9130: 'Equinculture',
    9147: 'Mgmt',
    9238: 'Social Service',
    9254: 'Tourism',
    9500: 'Nursing',
    9556: 'Oral Hygiene',
    9670: 'Advertising & Marketing',
    9773: 'Journalism & Comm.',
    9853: 'Basic Education',
    9991: 'Mgmt (evening)'
}

COURSE_CATEGORY_MAPPING = {
    33: 1, 171: 2, 8014: 3, 9003: 1, 9070: 2, 9085: 3,
    9119: 2, 9130: 1, 9147: 4, 9238: 3, 9254: 4, 9500: 3,
    9556: 3, 9670: 4, 9773: 5, 9853: 6, 9991: 4
}

COURSE_CONDENSED_MAPPING = {
    1: 'Agriculture',
    2: 'Tech',
    3: 'Health', 
    4: 'Business',
    5: 'Journalism',
    6: 'Education'
}

FEATURE_GROUPS = {
    'Demographic Features': [
        'Marital status',
        'Age at enrollment',
        'International',
        'Displaced'
    ],
    'Application Features': [
        'Application mode',
        'Course_Category',
        'Course_Name',
        'Previous qualification (grade)',
        'Admission grade'
    ],
    'Academic Performance': [
        'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)',
        'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)',
        'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)',
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)',
        'first_sem_success_ratio',
        'second_sem_success_ratio',
        'average_grade',
        'performance_change'
    ],
    'Economic Factors': [
        'Scholarship holder',
        'Tuition fees up to date',
        'Debtor',
        'Unemployment rate',
        'Inflation rate',
        'GDP',
        'economic_factor'
    ]
}

def get_course_mapping():
    return {
        33: "Biofuel Prod",
        171: "Animation Design",
        8014: "Social Service EC",
        9003: "Agronomy",
        9070: "Communication Design",
        9085: "Veterinary Nursing",
        9119: "Informatics Engineering",
        9130: "Equinculture",
        9147: "Management",
        9238: "Social Service",
        9254: "Tourism",
        9500: "Nursing",
        9556: "Oral Hygiene",
        9670: "Advertising",
        9773: "Journalism",
        9853: "Education",
        9991: "Management EC"
    }
