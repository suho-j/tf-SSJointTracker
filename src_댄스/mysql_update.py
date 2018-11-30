import pymysql
import datetime

HOST = '127.0.0.1'
PORT = 3306
USER = 'root'
PASSWORD = '123123'
DATABASE = 'sstest'
TABLENAME = 'person1_1'


def SQL_EXECUTE(query_string, tablename = TABLENAME) : 
    try :
        result = None
        conn = pymysql.connect(host=HOST, port=PORT, user=USER, password=PASSWORD, database=DATABASE)
        if conn.open :
            with conn.cursor() as curs:             
                curs.execute(query_string)
                result = curs.fetchall()
            conn.commit()
        else :
            assert (not conn.open)
    finally :
        conn.close()
        return result

def SQL_insertArr(arr_X, arr_Y, arr_Z) :
    arr_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,18]
    # arr_index = [0,1,2,3,4,5,6,7,8,9,18]
    arr_return = []

    for index in arr_index :
        arr_return.append(str(arr_X[index]))
        arr_return.append(str(arr_Y[index]))
        arr_return.append(str(arr_Z[index]))
        
    return arr_return


def SQL_CREATETABLE(tablename = TABLENAME) :
    # distance values call (humans list length = people).
    # NOSE = 0
    # NECK = 1
    # RSHOULDER = 2
    # RELBOW = 3
    # RWRIST = 4
    # LSHOULDER = 5
    # LELBOW = 6
    # LWRIST = 7
    # RHIP = 8
    # RKNEE = 9
    # RANKLE = 10
    # LHIP = 11
    # LKNEE = 12
    # LANKLE = 13
    # PUBIS = 18
    query_string = ('create table ' + tablename + '('
                                                  ' ID varchar(30),'
                                                  ' NOSE_X float(10),        NOSE_Y float(10),      NOSE_Z float(10),'
                                                  ' NECK_X float(10),        NECK_Y float(10),      NECK_Z float(10),'
                                                  ' RSHOULDER_X float(10),   RSHOULDER_Y float(10), RSHOULDER_Z float(10),'
                                                  ' RELBOW_X float(10),      RELBOW_Y float(10),    RELBOW_Z float(10),'
                                                  ' RWRIST_X float(10),      RWRIST_Y float(10),    RWRIST_Z float(10),'
                                                  ' LSHOULDER_X float(10),   LSHOULDER_Y float(10), LSHOULDER_Z  float(10),'
                                                  ' LELBOW_X float(10),      LELBOW_Y float(10),    LELBOW_Z float(10),'
                                                  ' LWRIST_X float(10),      LWRIST_Y float(10),    LWRIST_Z float(10),'
                                                  ' RHIP_X float(10),        RHIP_Y float(10),      RHIP_Z float(10),'
                                                  ' RKNEE_X float(10),       RKNEE_Y float(10),     RKNEE_Z float(10),'
                                                  ' RANKLE_X float(10),      RANKLE_Y float(10),    RANKLE_Z float(10),'
                                                  ' LHIP_X float(10),        LHIP_Y float(10),      LHIP_Z float(10),'
                                                  ' LKNEE_X float(10),       LKNEE_Y float(10),     LKNEE_Z float(10),'
                                                  ' LANKLE_X float(10),      LANKLE_Y float(10),    LANKLE_Z float(10),'
                                                  ' PUBIS_X float(10),       PUBIS_Y float(10),     PUBIS_Z float(10),'
                                                  ' PRIMARY KEY (id)'
                                                  ')')
    SQL_EXECUTE(query_string, tablename)


def SQL_DROPTABLE(tablename = TABLENAME) :
    query_string = 'DROP TABLE ' + tablename + ';'
    SQL_EXECUTE(query_string, tablename)


# def SQL_UPDATE() : ??
# def SQL_DELETE() : ??

# FIX!
def SQL_DELETEALL(tablename = TABLENAME) :
    # query_string = 'INSERT INTO ' + tablename + ' VALUES ("' + getCurrentTime() + '", %s);' % var_string
    query_string = 'DELETE from ' + tablename
    SQL_EXECUTE(query_string, tablename)


#test function
def SQL_INSERTLINE(arr_X, arr_Y, arr_Z, tablename = TABLENAME) :
    SQL_DELETEALL()
    SQL_INSERT(arr_X, arr_Y, arr_Z, tablename = TABLENAME)


def SQL_INSERT(arr_X, arr_Y, arr_Z, tablename = TABLENAME) :
    varlist = SQL_insertArr(arr_X, arr_Y, arr_Z) 
    var_string = ', '.join(varlist)
    # print("var_str : "+ var_string)
    query_string = 'INSERT INTO ' + tablename + ' VALUES ("' + getCurrentTime() + '", %s);' % var_string

    SQL_EXECUTE(query_string, tablename)

# return currenttime format : yyyy-mm-dd_hh:mm:ss.'ms''ms'
def getCurrentTime() :
    year  = str(datetime.datetime.now().year)
    month = str(datetime.datetime.now().month).zfill(2) 
    day   = str(datetime.datetime.now().day).zfill(2) 
    hour  = str(datetime.datetime.now().hour).zfill(2) 
    minute= str(datetime.datetime.now().minute).zfill(2) 
    sec   = str(datetime.datetime.now().second).zfill(2) 
    micsec   = str(datetime.datetime.now().microsecond)

    dateString = year + '-' + month + '-' +  day + '_' + hour + ':' + minute + ':' + sec + '.' + micsec
    # print("curTime : " + dateString)
    return str(dateString)

def SQL_SHOWTABLES(personnumber):
    query_string = 'SHOW TABLES'
    result = SQL_EXECUTE(query_string)

    k = 1
    i = 0
    tbname = "person" + str(k) + "_1"
    #tbname = "dance1_" + str(k)
    # print(result)
    # check the last table name and create new table
    while i < len(result):
        if result[i][0] == tbname:
            i = 0
            k += 1
            tbname = "person" + str(k) +"_1"
            #tbname = "dance1_" + str(k)
            #SQL_CREATETABLE(tbname)
            print(tbname)
        else:
            i += 1
    print(personnumber)
    for j in range(personnumber):
        j += 1
        tbname = "person" + str(k) + "_" + str(j)
        print(tbname)
        SQL_CREATETABLE(tbname)
    return k

# SQL_DROPTABLE()
#SQL_CREATETABLE()
#SQL_SHOWTABLES(3)