const express = require('express');
const cors = require('cors');
const cookieParser = require('cookie-parser');
const mongoose = require('mongoose');
var passport = require('passport')
var session = require('express-session')
require('./passport')(passport)


const uri = "mongodb://127.0.0.1:27017/practice";

require('dotenv').config();

const app = express();

app.use(cookieParser());
app.use(cors());
app.use(express.json());
mongoose.set('strictQuery', true)
mongoose.connect(uri,function(err,db){
  if (err) throw err;
  console.log("DataBase Connected")
});

const indexRouter = require('./routes/index')
const userRoute = require('./routes/user')

app.use(session({
  secret: 'some',
  resave: true,
  saveUninitialized: true
}));

app.use(passport.initialize());
app.use(passport.session()); // persistent login sessions

app.use('/',indexRouter)
app.use('/users', userRoute);

app.listen(8080, () => {
    console.log(`Server is running on port:`);
});

