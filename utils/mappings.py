from .preprocessing import load_data

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

def app_mode_mapping():
    return {
        1: '1st phase - gen.',
        2: 'ord. 612/93',
        5: '1st phase - Az. ',
        7: 'Higher courses ',
        10: 'ord. 854-B/99 ',
        15: 'Internat. (bach.) ',
        16: '1st phase - Madeira ',
        17: '2nd phase - gen. ',
        18: '3rd phase - gen. ',
        26: 'ord. 533-A/99 (1) ',
        27: 'ord. 533-A/99 (2) ',
        39: 'Age over 23 ',
        42: 'Transfer ',
        43: 'Course change ',
        44: 'Tech. diploma ',
        51: 'Change inst./course ',
        53: 'Short cycle diploma ',
        57: 'Change inst. (Intl.) '
    }

def marital_status_mapping():
    return {
        1: 'Single ',
        2: 'Married ',
        3: 'Widower ',
        4: 'Divorced ',
        5: 'Facto union ',
        6: 'Legally separated '
    }

def course_mapping():
    return {
        33: 'Biofuel Tech ',
        171: 'Animation & media ',
        8014: 'Social Serv. (NC) ',
        9003: 'Agronomy ',
        9070: 'Comm. Design ',
        9085: 'Vet. Nursing ',
        9119: 'Informatics Eng. ',
        9130: 'Equinculture ',
        9147: 'Mgmt ',
        9238: 'Social Service ',
        9254: 'Tourism ',
        9500: 'Nursing ',
        9556: 'Oral Hygiene ',
        9670: 'Marketing ',
        9773: 'Journalism ',
        9853: 'Education ',
        9991: 'Mgmt (NC) '
    }

def mother_qualification_mapping():
    return {
        1: 'HS - 12th Yr ',
        2: 'Univ. - Bachelor ',
        3: 'Univ. - Degree ',
        4: 'Univ. - Master ',
        5: 'Univ. - Doctorate ',
        6: 'Univ. - Freq. ',
        9: '12th Yr not done ',
        10: '11th Yr not done ',
        11: '7th Year (Old) ',
        12: 'Other - 11th Yr ',
        14: '10th Yr ',
        18: 'gen. commerce ',
        19: 'Basic Ed (3rd) ',
        22: 'Technical-prof.',
        26: '7th Yr Schooling ',
        27: '2nd Cycle HS ',
        29: '9th Yr not done ',
        30: '8th Yr ',
        34: 'Unknown ',
        35: 'Can’t read or write ',
        36: 'Can read, no 4th Yr ',
        37: 'Basic Ed (1st) ',
        38: 'Basic Ed (2nd) ',
        39: 'Technology Spec .',
        40: 'Univ. - Degree (2) ',
        41: 'Spec. Higher Ed ',
        42: 'prof. Higher Tech ',
        43: 'Mult. Masters ',
        44: 'Mult. Doctorate '
    }

def father_qualification_mapping():
    return {
        1: 'HS - 12th Yr ',
        2: 'Univ. - Bachelor ',
        3: 'Univ. - Degree ',
        4: 'Univ. - Master ',
        5: 'Univ. - Doctorate ',
        6: 'Univ. - Freq. ',
        9: '12th Yr not done ',
        10: '11th Yr not done ',
        11: '7th Year (Old) ',
        12: 'Other - 11th Yr ',
        13: '2nd Yr Comp. HS ',
        14: '10th Yr ',
        18: 'Commerce course ',
        19: 'Basic Ed (3rd) ',
        20: 'Comp. HS ',
        22: 'Technical-prof. ',
        25: 'Comp. HS - not done ',
        26: '7th Yr Schooling ',
        27: '2nd Cycle HS ',
        29: '9th Yr not done ',
        30: '8th Yr ',
        31: 'Admin & Commerce ',
        33: 'Accy. & Admin prog ',
        34: 'Unknown ',
        35: 'Can’t read or write ',
        36: 'Can read, no 4th Yr ',
        37: 'Basic Ed (1st ) ',
        38: 'Basic Ed (2nd ) ',
        39: 'Technological Spec. ',
        40: 'Univ. - Degree ',
        41: 'Spec. Higher Ed ',
        42: 'Prof. Higher Tech ',
        43: 'Mult. Master ',
        44: 'Mult. Doctorate '
    }

def mother_occupation_mapping():
    return {
        0: 'Student ',
        1: 'Directors/Managers ',
        2: 'Intellectual/Science ',
        3: 'Intermediate Tech ',
        4: 'Admin Staff ',
        5: 'Personal Security ',
        6: 'Farm/Skilled Agri. ',
        7: 'Skilled Construction ',
        8: 'Machine Operator ',
        9: 'Unskilled Worker ',
        10: 'Armed Forces ',
        90: 'Other Situation ',
        99: '(blank) ',
        122: 'Health prof. ',
        123: 'Teacher ',
        125: 'ICT Specialist ',
        131: 'Science/Eng. aid',
        132: 'Health aid ',
        134: 'Legal/Social/Sports ',
        141: 'Secretary ',
        143: 'accy./Fin. Services ',
        144: 'Admin Support ',
        151: 'Personal Service ',
        152: 'Sellers',
        153: 'Personal Care ',
        171: 'Construction ',
        173: 'Jeweler/Artisan ',
        175: 'Food/Clothing Serv. ',
        191: 'Cleaning Workers',
        192: 'Unskilled Agri/Fish ',
        193: 'Unskilled Transport ',
        194: 'Meal Prep '
    }

def father_occupation_mapping():
    return {
        0: 'Student ',
        1: 'Directors/Managers ',
        2: 'Intellectual/Science ',
        3: 'Intermediate Tech ',
        4: 'Admin Staff ',
        5: 'Security ',
        6: 'Farm/Skilled Agrc. ',
        7: 'Skilled Construction ',
        8: 'Machine Operator ',
        9: 'Unskilled Worker ',
        10: 'Armed Forces ',
        90: 'Other Situation ',
        99: '(blank) ',
        101: 'Military Officers ',
        102: 'Military Sergeants ',
        103: 'Other Armed Forces ',
        112: 'Directors Admin ',
        114: 'Trade Directors ',
        121: 'Science/Math/Engineer ',
        122: 'Health prof. ',
        123: 'Teacher ',
        124: 'Admin Specialist ',
        131: 'Science aid ',
        132: 'Health aid ',
        134: 'Legal/Social/Sports ',
        135: 'ICT Technician ',
        141: 'Secretary ',
        143: 'accy./Fin. Service ',
        144: 'Admin Support ',
        151: 'Personal Service ',
        152: 'Sellers ',
        153: 'Personal Care ',
        154: 'Protection/Security ',
        161: 'Skilled Agriculture ',
        163: 'Farmers/Fishermen ',
        171: 'Construction ',
        172: 'Metal Worker ',
        174: 'Electric/Electronics ',
        175: 'Food/Clothing sales ',
        181: 'Machine Operator ',
        182: 'Assembly Worker ',
        183: 'Vehicle Drivers ',
        192: 'Unskilled Agri/Fish ',
        193: 'Unskilled Transport ',
        194: 'Meal Prep ',
        195: 'Street Vendors '
    }



# Define the new category mapping for Course into 6 categories
def course_condensed_mapping():
    return {
        1: 'Agriculture ',
        2: 'Tech ',
        3: 'Health ',
        4: 'Business ',
        5: 'Journalism ',
        6: 'Education '
    }

# Map each specific course code to one of the 6 categories
def course_numeric_assignment(course):
    mapping = {
        33: 1,   # Agriculture (Biofuel Production Technologies)
        171: 2,  # Tech (Animation and Multimedia Design)
        8014: 3, # Health (Social Service evening)
        9003: 1, # Agriculture (Agronomy)
        9070: 2, # Tech (Communication Design)
        9085: 3, # Health (Veterinary Nursing)
        9119: 2, # Tech (Informatics Engineering)
        9130: 1, # Agriculture (Equinculture)
        9147: 4, # Business (Management)
        9238: 3, # Health (Social Service)
        9254: 4, # Business (Tourism)
        9500: 3, # Health (Nursing)
        9556: 3, # Health (Oral Hygiene)
        9670: 4, # Business (Advertising and Marketing Management)
        9773: 5, # Journalism (Journalism and Communication)
        9853: 6, # Education (Basic Education)
        9991: 4  # Business (Management evening)
    }
    return mapping.get(course, "Unknown")



# Step 1: Map the courses to their respective categories (1-6)
df = load_data()
df['Course_Category'] = df['Course'].map(course_numeric_assignment)

def category_dict():
    return {
        "marital_status_mapping": "Marital Status",
        "course_mapping": "Course",
        "father_occupation_mapping": "Father's Occupation",
        "mother_occupation_mapping": "Mother's Occupation",
        "father_qualification_mapping": "Father's Qualification",
        "mother_qualification_mapping": "Mother's Qualification",
        "app_mode_mapping": "Application Mode",
        "course_condensed_mapping": "Major"
    }