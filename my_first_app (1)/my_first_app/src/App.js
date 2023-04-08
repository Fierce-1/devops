import React from 'react';
import "bootstrap/dist/css/bootstrap.min.css";
import { BrowserRouter as Router, Routes , Route} from "react-router-dom";

import Login from "./components/Login.component";
import Signup from "./components/Signup.component";
import Dashboard from './components/dashboard.component';
import File from './components/file.component';
import Features from './components/features.component';
import Visual from './components/visual.component';

function App() {
  return (    
      <Router>
        <Routes>
          <Route path='/' element={<Login/>}></Route>
          <Route path='/signup' element={<Signup/>}></Route>
          <Route path='/dashboard' element={<Dashboard/>}></Route>
          <Route path='/file' element={<File/>}></Route>
          <Route path='/feature' element={<Features/>}></Route>
          <Route path='/vis' element={<Visual/>}></Route>
        </Routes>      
      </Router>
  );
}

export default App;