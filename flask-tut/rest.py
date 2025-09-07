
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# A todo list application

todos = [
    {"id": 1, "task": "Complete the course", "completed": False},
    {"id": 2, "task": "Wash clothes", "completed": True},
]


# Home route to serve the frontend
@app.route('/')
def home():
    return render_template("todos.html")


# GET: Retrieve all todos
@app.route('/todos', methods=['GET'])
def get_todos():
    return jsonify(todos)


# GET: Retrieve a todo by id
@app.route('/todos/<int:id>', methods=['GET'])
def get_todo_by_id(id):
    item = next((todo for todo in todos if todo["id"] == id), None)
    if item is None:
        return jsonify({"error": "Todo Not found"}), 404
    return jsonify(item)


# POST: Create a new todo
@app.route("/todos", methods=['POST'])
def add_todo():
    data = request.get_json()
    if not data or 'task' not in data:
        return jsonify({"error": "Task not found"}), 400
    new_item = {
        'id': todos[-1]["id"] + 1 if todos else 1,
        "task": data['task'],
        "completed": False
    }
    todos.append(new_item)
    return jsonify(new_item), 201

# PUT: Update a todo by id
@app.route('/todos/<int:id>', methods=['PUT'])
def update_todo(id): 
    data = request.get_json()
    item = next((todo for todo in todos if todo["id"] == id), None)
    if item is None:
        return jsonify({"error": "Todo Not found"}), 404
    if 'task' in data:
        item['task'] = data['task']
    if 'completed' in data:
        item['completed'] = data['completed']
    return jsonify(item)

# DELETE: Delete a todo by id
@app.route('/todos/<int:id>', methods=['DELETE'])
def delete_todo(id):
    global todos
    item = next((todo for todo in todos if todo["id"] == id), None)
    if item is None:
        return jsonify({"error": "Todo Not found"}), 404
    todos = [todo for todo in todos if todo["id"] != id]
    return jsonify({"result": True})


# Run the Flask app
if __name__ == "__main__":
    app.run(port=8000, debug=True)