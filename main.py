from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class Student(BaseModel):
    id: int
    name: str
    grade: int


students = [
    Student(id = 1, name= "ali", grade= 150), 
    Student(id = 2, name= "nesma", grade= 120),
            ]

@app.get("/students/")
def read_students():
    return students


@app.post("/students/")
def create_student(new_student: Student):
    students.append(new_student)
    return new_student

@app.put("/students/{id}")
def update_student(id: int, updated: Student):
    for index, student in enumerate(students):
        if student.id == id:
            students[index] = updated
            return updated
    return {"Error":"Error 404: Student not found"}

@app.delete("/students/{id}")
def delete_student(id: int):
    for index, student in enumerate(students):
        if student.id == id:
            del (students[index])
            return {"msg":"student deleted"}
    return {"Error":"Error 404: Student not found"}