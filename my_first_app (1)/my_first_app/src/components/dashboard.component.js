import {useState,useEffect} from 'react';
import { Navigate, Link, useLocation } from 'react-router-dom';
import '../stylesheet/dashboard.css';
import BarChartExample from '../charts/barchart';
import PieChartComponent from '../charts/piechart';
import DomainChart from '../charts/domainchart';
import PythonChart from '../charts/pythonGraph';
import {GaugeChartExample, GaugeChartExample2, GaugeChartExample3} from '../charts/guagechart';
import i3 from '../assets/i3.jpg';
import cytomate from '../assets/cytomate.png';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { library } from "@fortawesome/fontawesome-svg-core";
import {faChartBar, faKey, faBars, faUsers, faLocationCrosshairs, faFileImport, faSearch, faCircleExclamation ,faSignOut} from "@fortawesome/free-solid-svg-icons";
import Highlightedusers from '../charts/highlightedusers';
import MyTable from '../charts/table';
import jwtDecode from 'jwt-decode';
import Plot from 'react-plotly.js'

library.add(faChartBar, faKey, faBars, faUsers, faLocationCrosshairs, faFileImport, faSearch, faCircleExclamation, faSignOut);

function Dashboard() {
    const location = useLocation();
    const { state } = location;
    var eventDayWise = state && state.AllData;
    var plotData = state && state.AllData
    var TaskCateg = state && state.AllData;
    var highAnomalyUser = state && state.AllData;
    
    const [selectedUser, setSelectedUser] = useState(null);

    const handleUserChange = (event) => {
      setSelectedUser(event.target.value);
      console.log('User:',event.target.value)    
    };

   

    //This is graph for Events per Day
    if (state && state.AllData) {
        eventDayWise = state.AllData;
    } else {
      // use default data here
      eventDayWise =  {
        'Ali': {
          'eventperday': {
            'date': ['2016-08-25' ,'2016-08-26','2016-08-27','2016-08-28'],
            'total': [40, 208, 221, 156]
          }
        },
        'Ehtisham': {
          'eventperday': {
            'date': ['2016-08-25' ,'2016-08-26','2016-08-27','2016-08-28'],
            'total': [40, 20, 31, 156]
          }
        }
      };
    }
    
    //This is graph for TaskCategory Overall
    if (state && state.AllData) {
        TaskCateg = state.AllData;
       
    } else {
        // use default data here

        TaskCateg = {
            'Ali': {
              'taskCategoryOverall': {
                'category': ['Logon' ,'Logoff','Object Access','Account Management'], 
                'total': [20, 128, 121, 16]}
            },
            'Ehtisham': {
              'taskCategoryOverall': {
                'category': ['Logon' ,'Logoff','Object Access','Account Management'], 
                'total': [20, 128, 121, 16]}
              }
            }
          
    }
    
    //This is for python Data
    if (state && state.AllData) {
        if (selectedUser !== null){
            plotData =  JSON.parse(state.AllData[selectedUser]['name']);
          }
      } else {
        // use default data here
        plotData = JSON.parse('{"data":[{"name":"Test loss","x":["2016-08-26T17:28:30","2016-08-26T18:30:23","2016-08-26T19:31:18","2016-08-26T21:06:38","2016-08-26T22:08:36","2016-08-26T23:16:01"],"y":[0.34229214119374307,1.0606021975737114,0.34229214119374307,1.0932670349662241,1.0606021975737114,0.016731280406238412],"type":"scatter"},{"name":"Threshold","x":["2016-08-26T17:28:30","2016-08-26T18:30:23","2016-08-26T19:31:18","2016-08-26T21:06:38","2016-08-26T22:08:36","2016-08-26T23:16:01"],"y":[3.602207075074409,3.602207075074409,3.602207075074409,3.602207075074409,3.602207075074409,3.602207075074409],"type":"scatter"}],"layout":{"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"showlegend":true,"title":{"text":"Test loss vs. Threshold"}}}')
      }

    //This is for Highlighted user in Organiztaion Data
    if (state && state.AllData) {
        highAnomalyUser = state.highAnomalyUser;

    } else {
        // use default data here
        highAnomalyUser = { name: ['John','Mary', 'David','Sarah','Mark','Mike','Morgan'], anomalies: [70,55,40,37,28,15,0] };
    }
    var userOptions = ['Try User']
    if (state && state.AllData){
        userOptions = Object.keys(state.AllData).map(user => (
            <option value={user} key={user}>{user}</option>
          ));
    }
    
    
    const [isLogged, setIsLogged] = useState(true);

    const verifyJwt = (token, secret) => {
      try {
        const verifySignature = jwtDecode(token);
    
        return true;
      } catch (error) {
       
        return false;
      }
    }
    
    
    const setLogin = () => {
      const tok = localStorage.getItem("token");
      if (tok === null) {
        setIsLogged(false);
      }
      else{
        verifyJwt(tok,'secretkeyishere')
      }
    }
  
    useEffect(() => {
      setLogin()  
    }, []);
    
   
    if (isLogged === false) {
      return <Navigate to="/" />;
    }


    return (

        <div className='mainContainer'>

            <div className='flex-item1'>
                
                <img src={cytomate} alt='logohere'></img>
                <button>
                    <FontAwesomeIcon icon={faBars} size='2x' className='sidebarIcons'/>
                </button>
                <Link to="/file" className='Linkbutton'>
                    <FontAwesomeIcon icon={faUsers} size='2x' className='sidebarIcons'/>
                </Link>
                <button>
                    <FontAwesomeIcon icon={faLocationCrosshairs} size='2x' className='sidebarIcons'/>
                </button>
                <button>
                    <FontAwesomeIcon icon={faChartBar} size='2x' className='sidebarIcons'/>
                </button>
                <Link to="/" onClick={() => localStorage.removeItem("token")} className='Linkbutton'>
                    <FontAwesomeIcon icon={faSignOut} size='2x' className='sidebarIcons'/>
                </Link>


            </div>

            <div className='flex-container1'>
                
                <div className='flex-item2'>
                    <div className='box1'>
                        <div className='namebox1'>
                            <div className='screenName'>
                                <h1 className='textColor'>Dashboard</h1>
                            </div>
                        </div>
                        <div className='searchbox1'>
                            <div className="search-container">
                                <input type="text" className="search-field" placeholder="Search..." />
                                <button className="search-button">
                                    <FontAwesomeIcon icon={faSearch}/>
                                </button>
                            </div>
                        </div>
                        <div className='profilebox1'>
                            <div className='profileItems'>
                                <button className="buttonStyle">Home</button>
                                <button className="buttonStyle">Users</button>
                                <button className="buttonStyle">Report</button>
                                <img className='ProfilePic' src={i3} alt="Profileimage" />
                                <h4>User_Husnain</h4>
                            </div>
                        </div>
                    </div>

                    <div className='box2'>
                        <div className='highlightedDetailsbox2'>
                            <div className='detail_1box2'>
                                <div className='headlineboxes'>
                                    <h4 className='textColor'>Total Anomalies</h4>
                                    <GaugeChartExample3 value={0.2}/>
                                    {/* <div className='headlinelogovalue'>
                                        <FontAwesomeIcon icon={faCircleExclamation} size='3x' className='headlineIcons'/>
                                        <h2 className='textColor'>200</h2>
                                    </div> */}
                                </div>
                                <div className='headlineboxes'>
                                    <h4 className='textColor'>Total Events</h4>
                                    <GaugeChartExample2 value={0.2}/>
                                    {/* <div className='headlinelogovalue'>
                                        <FontAwesomeIcon icon={faCircleExclamation} size='3x' className='headlineIcons'/>
                                        <h2 className='textColor'>200</h2>
                                    </div> */}
                                </div>
                                <div className='headlineboxes'>
                                    <h4 className='textColor'>Total Users</h4>
                                    <GaugeChartExample value={0.2}/>
                                    {/* <div className='headlinelogovalue'>
                                        <FontAwesomeIcon icon={faCircleExclamation} size='3x' className='headlineIcons'/>
                                        <h2 className='textColor'>200</h2>
                                    </div> */}
                                </div>
                                <div className='headlineboxes'>
                                    <h4 className='textColor'>Mitre Alerts</h4>
                                    <GaugeChartExample2 value={0.2}/>
                                    {/* <div className='headlinelogovalue'>
                                        <FontAwesomeIcon icon={faCircleExclamation} size='3x' className='headlineIcons'/>
                                        <h2 className='textColor'>200</h2>
                                    </div> */}
                                </div>
                            </div>
                            <div className='detail_2box2'>
                                <div className='graphdetail'>
                                    
                                    <div>
                                         <BarChartExample data={TaskCateg} /> 
                                    </div>
                                    <h4 className='textColor'>Top Offense Categories</h4>
                                </div>
                                <div className='piechartdetail'>
                                    
                                    <PieChartComponent data={eventDayWise}/>
                                    <h4 className='textColor'>User Counts</h4>
                                </div> 
                            </div>
                        </div>
                        <div className='highlightedUserssbox2'>
                            <div className='userProgress'>
                               <Highlightedusers data={highAnomalyUser}/> 
                            </div>
                        </div>
                    </div>

                    {/*<div className='box3'>
                        helllo
                        <p >hello</p>
                    </div>
                    <div className='box4'></div> */}

                </div>

            </div>

            <div className='flex-container2'>
                <div className='c2Div1'>
                    
                    <select id="user-select" onChange={handleUserChange}>
                    <option value="">-- Please Select --</option>
                        {userOptions}
                    </select>
                    {plotData && <Plot data={plotData.data} layout={plotData.layout} />}
                </div>
                {/* <div className='c2Div2'>
                    <div className='tablediv'>
                        <MyTable/>
                    </div>
                    
                </div> */}
                <div className='c2Div3'>
                    <div>
                        <DomainChart/>
                    </div>
                    
                </div>
            </div>

            {/* <div className='flex-container3'>

            </div> */}

        </div>

    );
}

export default Dashboard;
