import sqlite3
import os


def create_restaurant_db(db_name):
    """Create a restaurant database with bookings table"""
    db_file = f"{db_name}.db"

    # Remove existing database if it exists
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Removed existing {db_file}")

    conn = sqlite3.connect(db_file)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            party_size INTEGER NOT NULL,
            customer_name TEXT,
            status TEXT DEFAULT 'confirmed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Add some sample bookings to make it realistic
    sample_bookings = [
        ("2025-07-25", "18:00", 2, "John Smith", "confirmed"),
        ("2025-07-25", "20:00", 4, "Alice Johnson", "confirmed"),
    ]

    conn.executemany(
        """
        INSERT INTO bookings (date, time, party_size, customer_name, status)
        VALUES (?, ?, ?, ?, ?)
    """,
        sample_bookings,
    )

    conn.commit()
    conn.close()
    print(f"✓ Created {db_file} with sample bookings")


def check_database_content(db_name):
    """Check what's in the database"""
    conn = sqlite3.connect(f"{db_name}.db")
    cursor = conn.execute("SELECT * FROM bookings")
    bookings = cursor.fetchall()
    print(f"  {db_name} has {len(bookings)} existing bookings")
    for booking in bookings:
        print(f"    - {booking[1]} {booking[2]} for {booking[3]} people")
    conn.close()


if __name__ == "__main__":
    print("Setting up restaurant databases...")

    # Create databases for all restaurants
    create_restaurant_db("sushi_maru")
    create_restaurant_db("tokyo_ramen")
    create_restaurant_db("takoyaki_taro")

    print("\nDatabase contents:")
    check_database_content("sushi_maru")
    check_database_content("tokyo_ramen")
    check_database_content("takoyaki_taro")

    print("\n✅ All databases created successfully!")
