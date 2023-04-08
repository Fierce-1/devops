import React, { useState } from 'react';
import axios from 'axios';
import { Link,Navigate   } from 'react-router-dom';
import '../stylesheet/App.css';
import '../stylesheet/signup.css';
import newpic from '../assets/newpic.jpg'
import cytomate from '../assets/cytomate.png';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { library } from "@fortawesome/fontawesome-svg-core";
import {faEnvelope, faKey, faUser, faLock } from "@fortawesome/free-solid-svg-icons";
library.add(faEnvelope, faKey, faUser, faLock);

function Signup() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [name, setname] = useState('');
  const [cpassword, setcPassword] = useState('');
  const [loggedIn, setLoggedIn] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();

    const user = {
    Company_Name:name,    
    username: username,
    password: password,
    };

    try {
        if(password === cpassword){
            const response = await axios.post('http://localhost:8080/users/signup', user);
            if (response.status===200) {
              alert("Created")
              setUsername('');
              setPassword('');
              setLoggedIn(true)
            } else {
              alert("Error")
            }   
        }
        else{
            alert('Check You Fields')
        }
    } catch (error) {
      alert(error)
    }
  };

  if (loggedIn) {
    console.log('here',loggedIn)
    return <Navigate to='/' />;
  }

  return (

    <div className='maindiv'>
        <div className='card'>
            <div className='div1sign'>
                <div className='loginstyle'>
                    <div className='welcomestyle'>
                        <img className='logostyle' src={cytomate} alt='logo here'></img>
                        <h3 className='text1'>Create Your Account</h3>
                    </div>
                    <form onSubmit={handleSubmit}>
                        <div className='formfields'>
                            <FontAwesomeIcon icon={faUser} color='#629B94'/>
                            <input className='inputfield'
                                type="text" 
                                name="username"  
                                placeholder = "company name"
                                required
                                onChange={(e) => setname(e.target.value)}
                                >
                            </input>
                        </div>
                        
                        <div className='formfields'>
                            <FontAwesomeIcon icon={faEnvelope} color='#629B94'/>
                            <input className='inputfield'
                                type="text" 
                                name="username"  
                                placeholder = "Email"
                                required
                                onChange={(e) => setUsername(e.target.value)}
                                >
                            </input>
                        </div>

                        <div className='formfields'>
                            <FontAwesomeIcon icon={faKey} color='#629B94'/>
                            <input className='inputfield' 
                                type="password" 
                                name="username"  
                                placeholder = "Password"
                                required
                                autocomplete="new-password"
                                onChange={(e) => setPassword(e.target.value)}
                                >
                            </input>
                        </div>

                        <div className='formfields'>
                            <FontAwesomeIcon icon={faLock} color='#629B94'/>
                            <input className='inputfield' 
                                type="password" 
                                name="username"  
                                placeholder = "Confirm Password"
                                onChange={(e) => setcPassword(e.target.value)}
                                required>
                            </input>
                        </div>

                        <div className='checkin'>
                            <input type="checkbox"/>
                            <h6 className='text'>Agree to Term & Conditions</h6>
                        </div>
                        
                        <input className='loginbutton1' type="submit" value="Sign up"/>
                    </form>
                    <div className='signuplinkdiv'>
                    <h6 className='text1'>Don't have an account?</h6>
                          <Link to="/" style={{textDecoration:'none'}}><h6 className='text'>Login</h6></Link>
                    </div>

                </div>

            </div>

            <div className='div2'>
                <img src={newpic} alt="imagehere" ></img>
            </div>

        </div>

    </div>

);
}

export default Signup;
