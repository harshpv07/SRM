import pandas as pd 
import numpy as np 
import sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle 
import tika 
from tika import parser
import spacy

classes = np.array(['Advocate', 'Arts', 'Automation Testing', 'Blockchain',
       'Business Analyst', 'Civil Engineer', 'Data Science', 'Database',
       'DevOps Engineer', 'DotNet Developer', 'ETL Developer',
       'Electrical Engineering', 'HR', 'Hadoop', 'Health and fitness',
       'Java Developer', 'Mechanical Engineer',
       'Network Security Engineer', 'Operations Manager', 'PMO',
       'Python Developer', 'SAP Developer', 'Sales', 'Testing',
       'Web Designing'], dtype=object)

def predictProfile(sentence):
    vectorizer = TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=1500)
    profiles = list(classes)
    s = vectorizer.transform(sentence)
    load_model = pickle.load(open('./knnpickle_file' , 'rb'))
    num = load_model.predict(s)[0]
    return profiles[num]


def parser_file(input_file):
    tika.initVM()
    file_data = parser.from_file(input_file)
    text = file_data["content"]
    text = text + "1234567891"
    extracted_text = {}
    nlp = spacy.load('en_core_web_sm')
    nlp_text = nlp(text)
    tokens = [token.text for token in nlp_text if not token.is_stop]

    skills = ["machine learning",
             "deep learning",
             "nlp",
             "natural language processing",
             "mysql",
             "sql",
             "django",
             "computer vision",
              "tensorflow",
             "opencv",
             "mongodb",
             "artificial intelligence",
             "ai",
             "flask",
             "robotics",
             "data structures",
             "python",
             "c++",
             "matlab",
             "css",
             "html",
             "github",
             "php",
              "Java", "Spring Framework", "Hibernate", "RESTful APIs", "SQL", "Maven/Gradle", "Git",
 "Testing methodologies", "Test planning", "Test automation", "Selenium", "JUnit", "TestNG", "Cucumber", "Jenkins",
 "Linux/Unix", "Shell scripting", "Continuous integration/Continuous deployment (CI/CD) pipelines", "Ansible", "Puppet", "Chef", "Docker","Kubernetes", "AWS","Azure","GCP", "Prometheus", "ELK stack",
 "Python", "Django/Flask", "SQL/NoSQL databases", "RESTful APIs", "Git",
 "HTML", "CSS", "JavaScript", "Responsive design", "UI/UX principles", "Adobe Photoshop/Illustrator", "Bootstrap",
 "Human resources management", "Recruitment and selection", "Employee relations", "Performance management", "Training and development", "HRIS",
 "Hadoop ecosystem (HDFS, MapReduce, Hive, Pig, HBase, Spark)", "Big data technologies", "SQL/NoSQL databases", "Java/Python",
 "Blockchain fundamentals", "Smart contracts", "Distributed ledger technology (DLT)", "Cryptography", "Solidity", "Ethereum",
 "ETL (Extract, Transform, Load) tools (e.g., Informatica, Talend, SSIS)", "SQL/NoSQL databases", "Data warehousing concepts", "Scripting languages (e.g., Python, Shell)",
 "Operations management", "Process improvement", "Project management", "Team leadership", "Budgeting and forecasting", "Stakeholder management",
 "Python/R", "Machine learning algorithms", "Matplotlib", "Seaborn", "Tableau", "Statistical analysis", "Data preprocessing", "Big data technologies",
 "Sales techniques", "Customer relationship management (CRM) systems", "Negotiation skills", "Market analysis", "Lead generation", "Closing deals",
 "Engineering design principles", "AutoCAD","SolidWorks", "Manufacturing processes", "Materials science", "Thermodynamics", "Finite Element Analysis (FEA)",
 "Fine arts", "Visual arts", "Performing arts", "Art history", "Creativity", "Artistic expression",
 "MySQL","PostgreSQL","Oracle", "SQL queries", "Database design", "Database administration",
 "Circuit theory", "Power systems", "Control systems", "Electronics", "Renewable energy", "PLC programming",
 "Fitness training", "Nutrition", "Wellness coaching", "Exercise physiology", "Anatomy and physiology", "Sports medicine",
 "Project management methodologies (e.g., Agile, Waterfall)", "Project planning and scheduling", "Risk management", "Stakeholder communication", "PMO tools (e.g., Microsoft Project, JIRA)",
 "Requirement gathering and analysis", "Business process modeling", "Stakeholder management", "Documentation", "UML diagrams", "Agile methodologies",
 "C#", ".NET framework", "ASP.NET MVC/Core", "Entity Framework", "SQL Server", "RESTful APIs",
 "Selenium WebDriver","Appium","Cypress", "TestNG/JUnit", "BDD/TDD", "Continuous integration tools (e.g., Jenkins, Travis CI)",
 "Network protocols","TCP/IP","HTTP","SSLo7", "Firewall configuration", "Intrusion detection/prevention systems (IDS/IPS)", "VPN technologies", "Security best practices", "Penetration testing",
 "SAP modules (e.g., SAP ECC, SAP S/4HANA)", "ABAP programming language", "SAP Fiori/UI5", "SAP integration technologies", "SAP HANA database",
 "Civil engineering principles", "Structural analysis and design", "Construction management", "AutoCAD/Civil 3D", "Building codes and regulations", "Geotechnical engineering",
 "Legal research", "Case management", "Courtroom litigation", "Legal writing", "Client counseling", "Negotiation skills"
    ]

    skillset = []

    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    # check for bi-grams and tri-grams (example: machine learning)
    for token in nlp_text.noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)

    skills = [i.capitalize() for i in set([i.lower() for i in skillset])]

    extracted_text["Skills"] = skills 
    return predictProfile(skills)




if __name__ == "__main__":
    print(parser_file("./Harsh_Resume_SWE.pdf")) 
