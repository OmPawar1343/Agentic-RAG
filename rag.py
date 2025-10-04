import re
import difflib
from chromadb import PersistentClient
from embeddings.embedding_model import get_embedding_model
from langchain_community.llms.ollama import Ollama
import warnings
warnings.filterwarnings("ignore")
from tabulate import tabulate
import pandas as pd

# Initialize LLM
llm = Ollama(model="gemma:2b")

# Initialize ChromaDB
client = PersistentClient(path="db/chroma_db")
collection = client.get_collection(name="csv_collection")

# Embedding model
embedding_model = get_embedding_model()

# Column list
COLUMNS = [
    "EmpID", "FirstName", "LastName", "StartDate", "ExitDate", "Email", "BusinessUnit",
    "EmployeeStatus", "EmployeeType", "PayZone", "EmployeeClassificationType", "TerminationType",
    "TerminationDescription", "DepartmentType", "Division", "Date of Birth", "State",
    "JobFunctionDescription", "GenderCode", "LocationCode", "RaceDesc", "MaritalDesc",
    "Performance Score", "Current Employee Rating", "Title", "Supervisor", "FullName"
]

# Column synonyms for flexible user queries
COLUMN_SYNONYMS = {
    # EmpID
    "empid": "EmpID",
    "employee id": "EmpID",
    "id": "EmpID",
    
    # Names
    "firstname": "FirstName",
    "first name": "FirstName",
    "lastname": "LastName",
    "last name": "LastName",
    "full name": "FullName",
    "name": "FullName",
    "first name last name": "FullName",
    "firstname lastname": "FullName",
    
    # Dates
    "startdate": "StartDate",
    "start date": "StartDate",
    "joining date": "StartDate",
    "hire date": "StartDate",
    "exitdate": "ExitDate",
    "exit date": "ExitDate",
    "termination date": "ExitDate",
    "date of birth": "Date of Birth",
    "dob": "Date of Birth",
    "birth date": "Date of Birth",
    
    # Job / Title
    "title": "Title",
    "job title": "Title",
    "position": "Title",
    "role": "Title",
    
    # Supervisor
    "supervisor": "Supervisor",
    "manager": "Supervisor",
    "boss": "Supervisor",
    
    # Email
    "email": "Email",
    "email address": "Email",
    "work email": "Email",
    
    # Business / Department
    "businessunit": "BusinessUnit",
    "business unit": "BusinessUnit",
    "department": "BusinessUnit",
    "departmenttype": "DepartmentType",
    "department type": "DepartmentType",
   
    # Employee Status / Type
    "employeestatus": "EmployeeStatus",
    "employee status": "EmployeeStatus",
    "status": "EmployeeStatus",
    "employeetype": "EmployeeType",
    "employee type": "EmployeeType",
    "type": "EmployeeType",
    "payzone": "PayZone",
    "pay zone": "PayZone",
   
   # Employee Classification
    "employeeclassificationtype": "EmployeeClassificationType",
    "employee classification type": "EmployeeClassificationType",
    "classification": "EmployeeClassificationType",
    
    # Termination
    "terminationtype": "TerminationType",
    "termination type": "TerminationType",
    "termination reason": "TerminationType",
    "terminationdescription": "TerminationDescription",
    "termination description": "TerminationDescription",
    "termination details": "TerminationDescription",
    
    # Division
    "division": "Division",
    
    # Location / State
    "state": "State",
    "location state": "State",
    "location": "LocationCode",
    "location code": "LocationCode",
    "office": "LocationCode",
   
    # Job Function
    "jobfunctiondescription": "JobFunctionDescription",
    "job function description": "JobFunctionDescription",
    "job function": "JobFunctionDescription",
    "role function": "JobFunctionDescription",
    "role":  "JobFunctionDescription",
   
    # Gender
    "gender": "GenderCode",
    "gender code": "GenderCode",
    "sex": "GenderCode",
   
    # Race / Ethnicity
    "racedesc": "RaceDesc",
    "race": "RaceDesc",
    "ethnicity": "RaceDesc",
  
    # Marital
    "maritaldesc": "MaritalDesc",
    "marital status": "MaritalDesc",
    "marital": "MaritalDesc",
    "marriage": "MaritalDesc",
   
    # Performance
    "performancescore": "Performance Score",
    "performance score": "Performance Score",
    "performance": "Performance Score",
   
    # Current employee rating
    "current employeerating": "Current Employee Rating",
    "current employee rating": "Current Employee Rating",
    "employee rating": "Current Employee Rating",
    "current rating": "Current Employee Rating",
    "rating": "Current Employee Rating"
}

# Enhanced COLUMN_SYNONYMS for ambiguous and missing fields
COLUMN_SYNONYMS.update({
    "dob": "Date of Birth",
    "date of birth": "Date of Birth",
    "department": "DepartmentType",
    "businessunit": "BusinessUnit",
    "division": "Division",
    "job function": "JobFunctionDescription",
    "performance score": "Performance Score",
    "current employee rating": "Current Employee Rating",
})

# Normalize column based on synonyms, prefer longer phrases first, with fuzzy matching
def normalize_column(question: str):
    q = question.lower().replace("_", " ").replace("-", " ")
    for synonym in sorted(COLUMN_SYNONYMS.keys(), key=len, reverse=True):
        if synonym in q:
            return COLUMN_SYNONYMS[synonym]
    possible = [col.lower().replace("_", " ").replace("-", " ") for col in COLUMNS]
    matches = difflib.get_close_matches(q, possible, n=1, cutoff=0.7)
    if matches:
        idx = possible.index(matches[0])
        return COLUMNS[idx]
    return None

# Parse multiple columns
def parse_columns(question: str):
    q = question.lower()
    found = []
    for synonym, col in COLUMN_SYNONYMS.items():
        if synonym in q and col not in found:
            found.append(col)
    return found

# Parse question (updated for multiple employees)
def parse_question(question: str):
    qlow = question.lower()
    
    # detect "all details" query
    all_details = any(kw in qlow for kw in ["all details", "everything", "complete info", "full details"])
    
    # detect requested columns
    columns = parse_columns(question)
    
    # find all employee IDs (3–6 digit numbers)
    empids = re.findall(r"\b\d{3,6}\b", question)

    query_cleaned= question.lower()
    for synonym in COLUMN_SYNONYMS.keys():
        query_cleaned = query_cleaned.replace(synonym,'')

    reserved_words = ["dob", "empid", "all details", "everything", "complete info", "full details", "and", "of", ","]
    for word in reserved_words:
        query_cleaned = query_cleaned.replace(word, '')
    
    
    
    
    names = re.findall(r"\b([A-Za-z][A-Za-z'`-]+)\s+([A-Za-z][A-Za-z'`-]+)\b", query_cleaned)
    names = [f"{fn.title()} {ln.title()}" for fn, ln in names]
    
    return {
        "EmpIDs": empids,
        "Names": names,
        "Columns": columns,
        "AllDetails": all_details,
    }

# Extract actual values from context
def extract_values_from_context(context, columns):
    result = {}
    for col in columns:
        pattern = re.compile(rf"{col}:\s*(.*)", re.IGNORECASE)
        match = pattern.search(context)
        if match and match.group(1).strip():
            result[col] = match.group(1).strip()
        else:
            result[col] = "Not available"
    return result

# RAG query function (multi-employee support)
def rag_query(question: str, top_k=1):
    parsed = parse_question(question)
    empids, names, columns, all_details = parsed["EmpIDs"], parsed["Names"], parsed["Columns"], parsed["AllDetails"]

    if not empids and not names:
        empids, names = [], []

    all_answers = []

    # loop through empids
    for empid in empids:
        where_filter = {"EmpID": int(empid)}
        query_text = f"Retrieve data for employee {empid}"
        query_embedding = embedding_model.encode([query_text])[0]

        results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k, where=where_filter)
        if not results['documents'] or not results['documents'][0]:
            all_answers.append(f"## Employee {empid}\nNo data found.")
            continue

        context = results['documents'][0][0]

        if all_details or not columns:
            prompt = f"""
You are given context about a single employee:

{context}

Provide all details about this employee. Format as "Column: Value" for each field.
"""
            answer = llm.invoke(prompt)
        else:
            extracted = extract_values_from_context(context, columns)
            answer_lines = ["| Column | Value |", "|---|---|"]
            for col, val in extracted.items():
                answer_lines.append(f"| {col} | {val} |")
            answer = "\n".join(answer_lines)

        all_answers.append(f"## Employee {empid}\n{answer}")

    # loop through names
    for name in names:
        query_text = f"Retrieve data for employee named {name}"
        query_embedding = embedding_model.encode([query_text])[0]

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where={"FullName": name}
        )

        if not results['documents'] or not results['documents'][0]:
            all_answers.append(f"## Employee {name}\nNo data found.")
            continue

        context = results['documents'][0][0]

        if all_details or not columns:
            prompt = f"""
You are given context about a single employee:

{context}

Provide all details about this employee. Format as "Column: Value" for each field.
"""
            answer = llm.invoke(prompt)
        else:
            extracted = extract_values_from_context(context, columns)
            answer_lines = ["| Column | Value |", "|---|---|"]
            for col, val in extracted.items():
                answer_lines.append(f"| {col} | {val} |")
            answer = "\n".join(answer_lines)

        all_answers.append(f"## Employee {name}\n{answer}")

    return "\n\n".join(all_answers)


def markdown_to_table(md_text):
    sections = md_text.split("## ")
    output = []

    for sec in sections:
        if not sec.strip():
            continue

        lines = sec.split("\n")
        header = lines[0].strip()
        table_lines = [line.strip() for line in lines[1:] if "|" in line]

        if not table_lines:
            output.append(f"## {header}\n(No data found)")
            continue

        data = [line.split("|")[1:-1] for line in table_lines if "---" not in line]
        if not data:
            output.append(f"## {header}\n(No data found)")
            continue

        df = pd.DataFrame(data[1:], columns=[c.strip() for c in data[0]])

        if "FullName" in df["Column"].values:
            idx = df.index[df["Column"] == "FullName"].tolist()
            if idx:
                fullname_row = idx[0]
                if not df.at[fullname_row, "Value"].strip():
                    try:
                        first = df.loc[df["Column"] == "FirstName", "Value"].values[0]
                        last = df.loc[df["Column"] == "LastName", "Value"].values[0]
                        df.at[fullname_row, "Value"] = f"{first} {last}".strip()
                    except:
                        df.at[fullname_row, "Value"] = "Not available"

        output.append(f"## {header}\n" + tabulate(df, headers="keys", tablefmt="grid"))

    return "\n\n".join(output)


if __name__ == "__main__":
    print("👋 Welcome to the Employee CSV RAG Chatbot! 🗂️💬 (type 'quit' or 'exit' to stop)")
    while True:
        question = input("\nYour Question: ").strip()
        if question.lower() in ["quit", "exit"]:
            print("Thank You Goodbye! 👋")
            break

        print("🤔 Thinking... finding the best answer, please wait...")

        response = rag_query(question)

        try:
            print("\nAnswer:\n", markdown_to_table(response))
        except:
            print("\nAnswer:\n", response)
