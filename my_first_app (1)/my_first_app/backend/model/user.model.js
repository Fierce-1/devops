const mongoose = require('mongoose')
const bcrypt = require('bcrypt-nodejs')
const Schema = mongoose.Schema
const userSchema = new Schema({
    username : {
        type : String,
        required : true,
        unique : true,
        minlength : 3,
        trim : true
    },
    Company_Name : {
        type : String,
        required : true,
        unique : true,
        minlength : 3,
        trim : true
    },
    password : {
        type : String,
        required : true,
        minlength : 6
    }
},
    {
    timestamp :true
})

userSchema.methods.hashPassword = function(password){
    return bcrypt.hashSync(password,bcrypt.genSaltSync(10))
}
userSchema.methods.comparePassword = function(password,hash){
    return bcrypt.compareSync(password,hash)
}
const User = mongoose.model("users",userSchema);
module.exports = User;