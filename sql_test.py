import sqlite3

# Connecting to sqlite database
connection_obj = sqlite3.connect('zus_outlets.db')

# cursor object
cursor_obj = connection_obj.cursor()

# to select all column we will use
statement = '''SELECT day FROM outlets WHERE store_name = 'ZUS Coffee – Temu Business Centre City Of Elmina' LIMIT 10'''

cursor_obj.execute(statement)

print("Only one data")
output = cursor_obj.fetchall()
print(output)

connection_obj.commit()

# Close the connection
connection_obj.close()