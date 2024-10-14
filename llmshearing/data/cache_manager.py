import json
import os
from datetime import datetime

from joblib import Memory
from nicegui import ui

# Initialize joblib Memory
cache_dir = "/nvmefs1/daranhe/llm-shearing/out/joblib_cache"
memory = Memory(location=cache_dir, verbose=0)


def get_cache_data():
    data = []
    cache_dir = memory.location
    for root, dirs, files in os.walk(cache_dir):
        for dir in dirs:
            if dir != "func_code.py":
                func_path = os.path.join(root, dir)
                for hash_dir in os.listdir(func_path):
                    hash_path = os.path.join(func_path, hash_dir)
                    if os.path.isdir(hash_path):
                        metadata_file = os.path.join(hash_path, "metadata.json")
                        output_file = os.path.join(hash_path, "output.pkl")

                        if os.path.exists(metadata_file) and os.path.exists(
                            output_file
                        ):
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)

                            data.append(
                                {
                                    "function": dir,
                                    "hash": hash_dir,
                                    "timestamp": datetime.fromtimestamp(
                                        metadata["time"]
                                    ).strftime("%Y-%m-%d %H:%M:%S"),
                                    "duration": f"{metadata['duration']:.4f}s",
                                    "size": f"{os.path.getsize(output_file) / 1024:.2f} KB",
                                    "path": hash_path,
                                }
                            )
    data.sort(key=lambda x: x["timestamp"])
    return data


def clear_cache_item(item):
    os.remove(os.path.join(item["path"], "output.pkl"))
    os.remove(os.path.join(item["path"], "metadata.json"))
    os.rmdir(item["path"])
    print("Cache item cleared")
    table.update()


def create_table():
    columns = [
        {
            "name": "function",
            "label": "Function",
            "field": "function",
            "sortable": True,
        },
        {"name": "hash", "label": "Hash", "field": "hash"},
        {
            "name": "timestamp",
            "label": "Timestamp",
            "field": "timestamp",
            "sortable": True,
        },
        {
            "name": "duration",
            "label": "Duration",
            "field": "duration",
            "sortable": True,
        },
        {"name": "size", "label": "Size", "field": "size", "sortable": True},
        {"name": "actions", "label": "Actions", "field": "actions"},
    ]

    with ui.table(columns=columns, rows=get_cache_data()).classes("w-full") as table:
        table.add_slot(
            "body-cell-actions",
            """
            <q-td :props="props">
                <q-btn @click="$parent.$emit('clear', props.row)" icon="delete" flat dense color='red'/>
            </q-td>
        """,
        )

        def on_clear(e):
            clear_cache_item(e.args)

        table.on("clear", on_clear)

    return table


@ui.page("/")
def main():
    ui.label("Joblib Memory Cache Manager").classes("text-h4 q-mb-md")

    global table
    table = create_table()

    def refresh_table():
        table.rows = get_cache_data()
        table.update()

    ui.button("Refresh", on_click=refresh_table).classes("q-mt-md")


ui.run(port=8080)
