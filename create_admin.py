""" Apache License 2.0 Copyright (c) 2022 Pavel Bystrov
    create admin user utility """
import sys
from getpass import getpass
from webapp import create_app
from webapp.db import DB
from webapp.user.models import User
from webapp.stat.models import Author


def save_author(name):
    """ save author function """
    max_id = 0
    if DB.session.query(DB.func.count(Author.id)).scalar() > 0:
        max_id = DB.session.query(DB.func.max(Author.id)).scalar()
        max_id = int(max_id)
    new_author = Author(id=(max_id + 1), name=name)
    DB.session.add(new_author)
    DB.session.commit()
    return max_id + 1


if __name__ == "__main__":
    # main method
    app = create_app()
    with app.app_context():
        username = input("Введите имя пользователя: ")

        if User.query.filter(User.username == username).count():
            print("Такой пользователь уже есть")
            sys.exit(0)

        password = getpass("Введите пароль: ")
        password2 = getpass("Повторите пароль: ")
        if not password == password2:
            print("password != password2")
            sys.exit(0)
        new_user = User(username=username, role="admin")
        new_user.set_password(password)
        DB.session.add(new_user)
        DB.session.commit()
        aid = save_author(username)
        print(f"User with id {new_user.id} added {aid}")
