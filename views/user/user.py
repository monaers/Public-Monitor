import bcrypt
from flask import Flask,session,render_template,redirect,Blueprint,request
from utils.query import query
from utils.errorResponse import errorResponse
import time
from datetime import datetime
from bcrypt import hashpw,gensalt,checkpw
ub = Blueprint('user',__name__,url_prefix='/user',template_folder='templates')
def checkpw(stored_password,provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password.encode('utf-8'))
@ub.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        username = request.form.get('username')
        password = request.form.get('password')
        def get_user_from_db(username):
            return query('select * from public.user where username = %s',[username],'select_one')
        users = get_user_from_db(username)
        if users and checkpw(users.get('password'),password):
            session['username'] = username
            return redirect('/page/home')
        else:
            return errorResponse('账号或密码错误')


@ub.route('/register',methods=['GET','POST'])
def register():
    if request.method == "GET":
        return render_template('register.html')
    else:
        if request.form['password'] != request.form['checkPassword']:
            return errorResponse('两次密码不同')
        def filter_fn(user):
            return request.form['username'] in user
        users = query('select * from public.user',[],'select')
        if users is None:
            return errorResponse('Database query failed or returned None')
        filter_list = list(filter(filter_fn,users))
        if len(filter_list):
            return errorResponse('该账户已被注册')
        else:
            hashed_password = hashpw(request.form['password'].encode('utf-8'),gensalt())
            str_hashed_password = hashed_password.decode('utf-8')
            time_tuple = time.localtime(time.time())
            query('''insert into "user"(username,password,creattime) values(%s,%s,%s)''',
                  (request.form['username'], str_hashed_password,
                   datetime(*time_tuple[:6]).strftime('%Y-%m-%d')))
        return redirect('/user/login')

