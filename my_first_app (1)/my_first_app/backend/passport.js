const localStrategy = require('passport-local').Strategy;
let User = require('./model/user.model');

module.exports = function(passport){
    passport.serializeUser(function(user,done){
        done(null,user)
    })
    passport.deserializeUser(function(user,done){
        done(null,user)
    })

    passport.use(new localStrategy(function(username,password,done){
        console.log(username)
        User.findOne({"username":username}).then(function(doc,err){
            if(err){ 
              done(err)
              console.log('Error Occured',err,'doc ',doc)
            }
            else{
                if(doc){
                    let valid = doc.comparePassword(password,doc.password)
                    if (valid){
                        console.log('Done',valid)
                        return done(null,doc)
                    }
                    else{
                        return done(null,false)
                    }
                }
                else{
                    return done(null,false)
                }
            }
        })
    }))
}