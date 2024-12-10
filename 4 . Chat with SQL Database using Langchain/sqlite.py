import sqlite3 

## Code to connect to sqlite3
connection=sqlite3.connect("student.db")


##Cursor object to insert record  , create table
cursor=connection.cursor()

##create table:
table_info="""
    create table Student(
        Name varchar(25),
        Class varchar(25),
        Section varchar(25) , 
        Marks int
    )
"""


## create table in database:
cursor.execute(table_info)


## insert the data into the sqlite 3 database
cursor.execute("""insert into Student values('krish' , 'Data Science' , 'A' , 90)""")
cursor.execute("""insert into Student values('John', 'Data Science', 'B', 100)""")
cursor.execute("""insert into Student values('Mukesh', 'Data Science', 'A', 86)""")
cursor.execute("""insert into Student values('Jacob', 'DEVOPS', 'A', 50)""")
cursor.execute("""insert into Student values('Dipesh', 'DEVOPS', 'A', 35)""")

## Display all the records:

print("Inserted records are:")

data=cursor.execute("""select * from Student""")

for row in data:
    print(row)
    
    
## commit your changes in the database
connection.commit()
connection.close()

