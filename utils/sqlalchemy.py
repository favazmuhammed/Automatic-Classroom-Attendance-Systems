from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker
from django.conf import settings

def create_engine_from_django_settings():
    db = settings.DATABASES['default']
    engine = db['ENGINE'].lower()
    

    if 'postgresql' in engine:
        sqlalchemy_engine = 'postgresql'
    else:
        raise NotImplementedError('Could not determine what engine to use for "%s" automatically' % engine)

    host = 'localhost:5432'
    name = db['NAME']
    credentials = [db['USER']]
    password = db['PASSWORD']
    if password:
        credentials.append(password)

    user_password = ':'.join(credentials)

    dsn = '{engine}+psycopg2://{user_password}@{host}/{name}'.format(
        engine=sqlalchemy_engine, user_password=user_password, host=host, name=name)

    return create_engine(dsn)


class SqlAlchemySession:
    def __init__(self):
        self.Session = sessionmaker(bind=create_engine_from_django_settings())

    def get_data(self,sql_stmnt):
        try:
            with self.Session() as session:
                result = session.execute(sql_stmnt)
                session.close()
            return result
        except Exception as err:
            raise RuntimeError(f'Failed to execute query -- {err}') from err

    def get_data_with_values(self,sql_stmnt,values):
        try:
            with self.Session() as session:
                result = session.execute(sql_stmnt,values)
                session.close()
            return result
        except Exception as err:
            raise RuntimeError(f'Failed to execute query -- {err}') from err