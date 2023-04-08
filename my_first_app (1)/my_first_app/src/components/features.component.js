import React, { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";
import "../stylesheet/feature.css";


function CheckboxPage() {
  const [selectedValues, setSelectedValues] = useState([]);
  const [singleUser, setSingleUser] = useState(false);
  const [multiUser, setMultiUser] = useState(false);
  const [selectDate, setSelectDate] = useState(false);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [wholeDate, setWholeDate] = useState(false);
  const [outcomeTime, setOutcomeTime] = useState(false);
  const [outcomeCount, setOutcomeCount] = useState(false);
  const [outcomePattern, setOutcomePattern] = useState(false);
  const [dropdownOptions, setDropdownOptions] = useState([]);
  //const [sendCharData,setSendCharData] = useState({date: [],total: []})
  
  const navigate = useNavigate();
  const location = useLocation();
  const { dropdownOptions: uniqueValues } = location.state;

  useState(() => {
    if (uniqueValues) {
      setDropdownOptions(uniqueValues);
    }
  }, [uniqueValues]);

  const handleWholeDateChange = (event) => {
    setWholeDate(event.target.checked);
    setSelectDate(false)
  };
  
  const handleSelectDateChange = (event) => {
    setSelectDate(event.target.checked);
    setWholeDate(false)
  };

  const handleStartDateChange = (event) => {
    if (event.target && event.target.value !== undefined) {
      setStartDate(event.target.value);
    }
  };
  
  const handleEndDateChange = (event) => {
    if (event.target && event.target.value !== undefined) {
      setEndDate(event.target.value);
    }
  };
  
  const handleSingleUserChange = () => {
    setSingleUser(true);
    setMultiUser(false);
  };
  
  const handleMultiUserChange = () => {
    setSingleUser(false);
    setMultiUser(true);
  };

  const handleOutcomeTimeChange = (event) => {
    setOutcomeTime(event.target.checked)
  };
  
  const handleOutcomeCountChange = (event) => {
    setOutcomeCount(event.target.checked)
  };

  const handleOutcomePatternChange = (event) => {
    setOutcomePattern(event.target.checked )
  };
  

  const handleDropdownChange = (event) => {
    const { value } = event.target;
    setDropdownOptions(dropdownOptions.filter((option) => option !== value));
    setSelectedValues([...selectedValues, value]);
  };

  const handleDeleteClick = (value) => {
    setSelectedValues(selectedValues.filter((v) => v !== value));
    setDropdownOptions([...dropdownOptions, value]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const formData = new FormData();
      if ((singleUser || multiUser) && (wholeDate || (startDate && endDate))  && (outcomeCount || outcomeTime || outcomePattern)){
        formData.append("singleUser", singleUser);
        formData.append("multiUser", multiUser);
        formData.append("wholeDate", wholeDate);
        formData.append("count", outcomeCount);
        formData.append("time", outcomeTime);
        formData.append("pattern", outcomePattern);
        if (!wholeDate) {
          formData.append("startDate", startDate);
          formData.append("endDate", endDate);
        }
        formData.append("Users", selectedValues);
        const response = await axios.post("http://localhost:5000/feature",  formData);
        
        if (response.status === 200) {
          const uniqueList = response.data; // Get the response data directly as JSON
          //setPlotData(JSON.parse(uniqueList));
          const keys = Object.keys(uniqueList);
          const anomlies = []
          const usersName = []
          for (let i = 0; i < keys.length; i++) {
            usersName.push(keys[i])
            anomlies.push(uniqueList[keys[i]]['count'])
          }
          
          for (let i = 0; i < keys.length; i++) {
            const key = keys[i];  
            navigate("/dashboard", { state: { AllData: uniqueList,highAnomalyUser :{name : usersName, anomalies : anomlies}}});
          }
          alert("Done") // navigate to next page
        }
        else{
          alert("Fail") // navigate to next page
        }
      }
      else{
        alert('Fill All fields');
      } 
    }
    catch(error) {
      console.error("Error: ", error);
    }
      
  };

  return (
    <div className="containerFeature">
      <h2>Details</h2>
      <form onSubmit={handleSubmit} className="form1">
        <div className="checkbox-container1">
          <div>
              <h3>Select User</h3>
              <label>
            <input
              type="checkbox"
              checked={singleUser}
              onChange={handleSingleUserChange}
            />
            Single User
            </label>
            <label>
              <input
                type="checkbox"
                checked={multiUser}
                onChange={handleMultiUserChange}
              />
              Multi User
            </label>
          </div>
          <div>
            <h3>Select Date</h3>
            <label>
              <input
                type="checkbox"
                checked={selectDate}
                onChange={handleSelectDateChange}
                />
              Select Date
            </label>
            {selectDate && (
              <>
                <label>
                  Start Date:
                  <input
                    type="date"
                    value={startDate}
                    onChange={handleStartDateChange}
                  />
                </label>
                <label>
                  End Date:
                  <input
                    type="date"
                    value={endDate}
                    onChange={handleEndDateChange}
                  />
                </label>
              </>
            )}
            <label>
              <input
                type="checkbox"
                checked={wholeDate}
                onChange={handleWholeDateChange}  
              />
              Whole Date
            </label>
          </div>
        </div>
        <div className="checkbox-container1">
        <div className="checkbox-container2">
          <h3>Outcome</h3>
          <label>
        <input
          type="checkbox"
          checked={outcomeCount}
          onChange={handleOutcomeCountChange}
        />
        Count
      </label>
      <label>
        <input
          type="checkbox"
          checked={outcomeTime}
          onChange={handleOutcomeTimeChange}
        />
        Time
      </label>
      <label>
        <input
          type="checkbox"
          checked={outcomePattern}
          onChange={handleOutcomePatternChange}
        />
        Pattern
      </label>
      </div>
      <div className="dropdown-container">
          <h3>Dropdown Options</h3>
          <select onChange={handleDropdownChange}>
            <option value="">Select an option</option>
            {dropdownOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
          
        </div>
        </div>
        <div className="table-container">
          <h3>Selected Options</h3>
          <table>
            <thead>
              <tr>
                <th>Option</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {selectedValues.map((value) => (
                <tr key={value}>
                  <td>{value}</td>
                  <td>
                    <button onClick={() => handleDeleteClick(value)}>
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <button type="submit">Next</button>
      </form>
      
    </div>
  );
}


export default CheckboxPage;
