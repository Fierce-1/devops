import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Link, Navigate } from 'react-router-dom';
import '../stylesheet/App.css';
import pic from '../assets/pic.jpeg';
import cytomate from '../assets/cytomate.png';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { library } from "@fortawesome/fontawesome-svg-core";
import {faEnvelope, faKey, faUser, faLock } from "@fortawesome/free-solid-svg-icons";
library.add(faEnvelope, faKey, faUser, faLock);

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loggedIn, setLoggedIn] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const user = { username, password };
    try {
      const response = await axios.post('http://localhost:8080/users/login', user);
      localStorage.setItem('token', response.data);
      if (response.data) {
        console.log('token',response.data)
        setLoggedIn(true);
      } else {
        alert('Invalid login credentials');
      }
    } catch (error) {
      console.log(error);
    }
  };
  useEffect(()=>{
    const tok = localStorage.getItem('token')
    if (tok !== null){
      console.log('toekin',tok , loggedIn)
      setLoggedIn(true)
    }
  },[])
 
  if (loggedIn) {
    console.log('here',loggedIn)
    return <Navigate to='/dashboard' />;
  }

  return (

    <div className='maindiv'>
            <div className='card'>
                <div className='div1'>
                    <div className='logo'>
                        <img className='logostyle1' src={cytomate} alt='logo here'></img>
                        <h1 className='name'>Cytomate</h1>
                    </div>
                    <div className='imagediv1'>
                        <div className='imagediv2'>
                            <img src={pic} ></img>
                        </div>
                    </div>
                </div>

                <div className='div2'>
                    <div className='loginstyle'>
                        <div className='welcomestyle'>
                        <h1 className='text1'>Welcome Back!</h1>
                        <h3 className='text1'>Sign in to Dashboard</h3>
                        </div>
                        <form onSubmit={handleSubmit}>

                          
                          <div className='formfields'>
                              <FontAwesomeIcon icon={faEnvelope} color='#629B94'/>
                              <input className='inputfield'
                                  type="text" 
                                  name="username"  
                                  placeholder = "Email"
                                  onChange={(e) => setUsername(e.target.value)}
                                  required
                                  >
                              </input>
                          </div>
                          <div className='formfields'>
                              <FontAwesomeIcon icon={faKey} color='#629B94'/>
                              <input className='inputfield' 
                                  type="password" 
                                  name="username"  
                                  placeholder = "Password"
                                  onChange={(e) => setPassword(e.target.value)}
                                  required
                                  >
                              </input>
                          </div>
                          <div className='checkin'>
                              <input type="checkbox" className='check' />
                              <h6 className='text1'>Remember Me</h6>
                          </div>
                            
                          <input className='loginbutton1' type="submit" value="Login"/>
                        </form>
                        <div className='signuplinkdiv'>
                            <h6 className='text1'>Don't have an account?</h6>
                          <Link to="/signup" style={{textDecoration:'none'}}><h6 className='text'>  Signup</h6></Link>  
                        </div>
                    

                    </div>
                </div>
            </div>

        </div>


);
};

export default Login;
