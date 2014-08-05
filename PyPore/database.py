# database.py
# Contact: Jacob Schreiber
#          jmschreiber91@gmail.com

'''
This module contains connectors between google and SQL databases.
Currently only read is supported for Google spreadsheets, while
everything is supported for the SQL databases. SQL use case here.

>>> db = Database( db="cheeses", user="jim", password="1hGaj29Kajh", 
        host="127.0.0.1")
>>> db.execute( "SELECT * FROM cheese_list" )
(( 'cheddar', 'CHE' ), ( 'american', 'AME', ), ( 'gruye', 'GRU' ))
>>> db.execute( "INSERT INTO cheese_list VALUES ('mozarella', 'MOZ')" )
>>> db.execute( "SELECT * FROM cheese_list" )
(( 'cheddar', 'CHE' ), ( 'american', 'AME', ), ( 'gruye', 'GRU' ),
    ('mozarella', 'MOZ'))
>>> db.execute( "DELETE FROM cheese_list WHERE name LIKE '%%'" )
>>> db.execute( "SELECT * FROM cheese_list" )
()

You can also get tables, which allow you to have an OO interface to that specific
table. Assuming that the previous commands did not occur, you can do the following:

>>> table = db.get_table( "cheese_list" )
>>> table.column_names
[ 'name', 'tag' ]
>>> table.column_types
[ 'varchar', 'varchar' ]
>>> table.read()
(( 'cheddar', 'CHE' ), ( 'american', 'AME', ), ( 'gruye', 'GRU' ))
>>> table.read( columns=["name"])
(( 'cheddar' ), ( 'american' ), ( 'gruye' ))
>>> table.read( columns=["name"], values=["am%"])
(( 'american' ))
>>> table.insert( values=('mozarella', 'MOZ') )
>>> table.read()
(( 'cheddar', 'CHE' ), ( 'american', 'AME', ), ( 'gruye', 'GRU' ),
    ('mozarella', 'MOZ'))
>>> table.delete( columns=["name"], values=["mozarella"] )
>>> table.read()
(( 'cheddar', 'CHE' ), ( 'american', 'AME', ), ( 'gruye', 'GRU' ))
>>> table.insert( values=('MOZ', 'mozarella'))
>>> table.read()
(( 'cheddar', 'CHE' ), ( 'american', 'AME', ), ( 'gruye', 'GRU' ),
    ('MOZ', 'mozarella'))
>>> table.delete( columns=["name"], values=["mozarella"] )
>>> table.read()
(( 'cheddar', 'CHE' ), ( 'american', 'AME', ), ( 'gruye', 'GRU' ),
    ('MOZ', 'mozarella'))
>>> table.delete( columns=["name"], values=["MOZ"] )
>>> table.read()
(( 'cheddar', 'CHE' ), ( 'american', 'AME', ), ( 'gruye', 'GRU' ))
>>> table.insert( columns=["tag", "name"], values=["MOZ", "mozarella"])
>>> table.read()
(( 'cheddar', 'CHE' ), ( 'american', 'AME', ), ( 'gruye', 'GRU' ),
    ('mozarella', 'MOZ'))
'''

import MySQLdb
import itertools as it

class Database( object ):
    '''
    Represents a SQL database. 
    '''

    def __init__( self, db, user, password, host ):
        '''
        Take in the credentials for the server and connect to it, connecting
        to a specific database on the server.
        '''

        self.db = MySQLdb.connect( host, user, password, db )
        self.cursor = self.db.cursor()

    def execute( self, statement ):
        '''
        Allows the user to execute a specific SQL command. If an error is
        raised, raise an error.
        '''

        self.cursor.execute( statement )
        self.db.commit()
        return self.cursor.fetchall()

    def get_table( self, table ):
        '''
        Make a table object for a certain table.
        '''

        return Table( self, table )


    def read_table( self, table, columns=None, values=None ):
        '''
        A wrapper allowing you to read a table.
        '''

        table = self.get_table( table )
        return table.read( columns=columns, values=values ) 

class Table( object ):
    '''
    Represents a table in the database. Allows you to query that table directly
    instead of looking at the database level.
    '''

    def __init__( self, db, name ):
        '''
        Store the name. We can't actually 'login' to a table, to we'll just
        only call from that table in the future.
        '''
        self.db = db
        self.name = name

    @property
    def columns( self ):
        return self.db.execute( "SHOW COLUMNS FROM {}".format( self.name ) )

    @property
    def column_type_dict( self ):
        return { key: value for key, value, _, _, _, _ in self.columns }

    @property
    def column_names( self ):
        return map( lambda x: x[0], self.columns )

    @property
    def column_types( self ):
        return map( lambda x: x[1], self.columns )

    def read( self, columns=None, values=None ):
        '''
        Read certain columns from the table, or all by default.
        '''

        query = "SELECT {} FROM {}".format( 
            ','.join( columns ) if columns else '*', self.name )
        if values:
            query += " WHERE {}".format( self._build_clauses( values, columns ) )

        return self.db.execute( query )

    def insert( self, values, columns=None ):
        '''
        Allows you to insert one row into the database. Assume the ordering
        is as specified in the database unless columns are specified, then
        use that ordering.
        '''

        self.db.execute( "INSERT INTO {} {} VALUES ({})".format(
            self.name, "({})".format( ','.join( columns ) ) if columns else "",
            ",".join( "'{}'".format( str(v) ) for v in values  ) )  )

    def delete( self, entry, columns=None ):
        '''
        Allows you to delete anything matching this entry.
        '''

        self.db.execute( "DELETE FROM {} WHERE {}".format(
            self.name, self._build_clauses( entry, columns ) ) )

    def _build_clauses( self, values, columns=None ):
        '''
        A private function which will take a tuple of values, ordered according
        to the column order in the database, and build an appropriate set of
        clauses including "IS NULL", "=", "LIKE", and quotations as appropriate.
        '''

        # If columns are provided, they may be looking fur a custom ordering
        # of values, so use that. Else, use the natural ordering
        columns = columns or self.column_names
        column_type_dict = self.column_type_dict

        # Store the clauses for later use.
        clauses = []

        # Iterate through the column-value pairs, assuming that if they gave
        # a column and not a value that they don't care what that value is.
        for column, value in it.izip_longest( columns, values ):
            column_type = self.column_type_dict[ column ]

            # If the entry is None, they don't care what it is and
            # thus use a wildcard
            if value is None:
                value = '*'

            # Remove any extra white space that may be there
            value = value.strip()

            # SQL NULL is the same as the string None, not the datatype None
            if value == "None": 
                clauses.append( "{} IS NULL".format( column ) )

            # If the cell type is a varchar..
            elif 'varchar' in column_type:
                if value[-1] != '*':  
                    # If they are not looking for a wild card, look for exact match  
                    clauses.append( "{} = '{}'".format( column, value ) )
                else:
                    # Otherwise, allow for wild card
                    clauses.append( "{} LIKE '%{}%'".format(column, value[:-1]))
            elif 'float' in column_type or 'int' in column_type:
                clauses.append( "{} = {}".format( column, value ) )

        return ' AND '.join( clauses ) or None

class GoogleSpreadsheet( object ):
    """
    Wrapper for gspread to connect to a google spreadsheet and read it easily.
    It acts as a generator, so use is as follows:

    gs = GoogleSpreadsheet( email, password, title )
    for row in gs:
        print row

    table = [ row for row in gs ]
    """

    def __init__( self, email, password, title, sheet="sheet1" ):
        """
        Connect to the database and open it.
        """

        import gspread
        self.gs = gspread.login( email, password )
        self.ws = self.gs.open( title ).worksheet( sheet )

    def read( self ):
        """
        Return the spreadsheet as a list of lists.
        """

        return self.ws.get_all_values()

class MySQLDatabaseInterface( object ):
    '''
    To use mySQL servers, must download the apporpriate servers. DEPRICATED.
    '''
    def __init__( self, db, user = None, password = None, host = None ):
        import MySQLdb
        self.db = MySQLdb.connect( host, user, password, db )
        self.cursor = self.db.cursor()

    def execute( self, statement ):
        '''
        Execute an arbitrary SQL statement. No restriction on the type of statements which
        can be executed, except those imposed by the SQL user. 
        '''
        try:
            self.cursor.execute( statement )
        except:
            raise DatabaseError( "MySQL Error: Unable to execute statement \
                '{}'".format(statement) )
        self.db.commit()

    def read( self, statement ):
        try:
            self.cursor.execute( statement )
            return self.cursor.fetchall()
        except:
            raise DatabaseError( "MySQL Error: Unable to execute statement \
                '{}'".format(statement) )

    def insert( self, table, data ):
        try:
            for row in data:
                self.cursor.execute( 'INSERT INTO {} VALUES ({})'.format( 
                    table, self._build_insert( row ) ) )
            self.db.commit()
        except:
            raise DatabaseError( "MySQL Error: Unable to add row ({}) \
                to table ({})".format(row, table ) )

    def _build_insert( self, tuple ):
        return ','.join( [ '"{}"'.format( str(item).replace('"', '""').replace( "\\", "\\\\") ) 
                            if isinstance( item, str ) or not item 
                            else '{}'.format( item ) 
                            for item in tuple ] )

    def _datify( self, date ):
        if not isinstance( date, Qc.QString ) and not isinstance( date, str ):
            return None
        if isinstance( date, datetime.date ):
            return date
        for seg in '/-':
            if date.count( seg ) == 2:
                date = date.split( seg )
        return datetime.date( int(date[0]), int(date[1]), int(date[2]) )

class DatabaseError( Exception ):
    def __init__( self, error ):
        self.error = error
    def __str__( self ):
        return repr( self.error ) 

