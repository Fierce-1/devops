const router = require('express').Router();
const jwt = require('jsonwebtoken')
const passport = require("passport");
require('dotenv').config()
let User = require('../model/user.model');
const multer = require("multer");
const upload = multer({ dest: "uploads/" });

router.route("/upload").post(upload.single("file"), (req, res) => {
  const file = req.file;
  console.log("File uploaded:", file.filename);
  res.send({ filename: file.filename });
});

router.route('/signup').post((req, res) => {
  const username = req.body.username;
  const password = req.body.password;
  const Company_Name = req.body.Company_Name;
  const newUser = new User();
  console.log('aaaaa',username)
  User.findOne({"username":username}).then(function(doc,err){
    if(err){ 
      res.status(500).send("Error Occured")
      console.log('Error Occured here',err,'doc ',doc)
    }

    else{
      if(doc){
        res.status(500).send("User Already Exist")
        console.log('Error Already',doc)
      }
      else{
          newUser.username = username
          newUser.password = newUser.hashPassword(password)
          newUser.Company_Name= Company_Name
          newUser.save()
          .then(() => {
            res.json('User added!')
          })
          .catch(err => {
            res.status(400).json('Error: ' + err)
            console.log('Error',err)
        });
      }
    }
  })
});

router.post('/login', passport.authenticate('local', { failureRedirect: '/' }), (req, res) => {
  console.log('here')
  jwt.sign({ id: req.user._id }, 'secretkeyishere', (err, token) => {
    if (err) {
      // Handle error
      console.error(err);
      return res.status(500).send('Internal Server Error');
    }

    // Set cookie with the JWT token
    res.cookie('token', token, {
      httpOnly: false,
      maxAge: 1000 * 60 * 60 * 24,
    });

    // Send success response
    res.status(200).send(token);
  });
});

function verifyToken(req,res,next){
  const header = req.headers.authorization;
  console.log('here in verify',header)
  if (header ){
    const tok = header.split(' ')
    const orgToken = tok[1]
    req.token=orgToken
    next()
  }
  else{
    res.sendStatus(403)
  }

}
module.exports = router;