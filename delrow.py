import sqlite3
conn = sqlite3.connect("data/db.sqlite3")
conn.execute("DELETE FROM group_admins WHERE chat_id=?", (609965713,))
conn.execute("DELETE FROM group_settings WHERE chat_id=?", (609965713,))
conn.commit()
conn.close()