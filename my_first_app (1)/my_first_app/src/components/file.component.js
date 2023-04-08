import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import ReactLoading from "react-loading";
import "../stylesheet/fileUpload.css";  

function FileUpload() {
  const [file, setFile] = useState(null); 
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://localhost:5000/upload", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
        console.log("File uploaded successfully");
        const uniqueList = await response.json(); // Parse the response as JSON
        console.log("File uploaded successfully",uniqueList);
        navigate("/feature", { state: { dropdownOptions: uniqueList } }); // navigate to next page
      } else {
        console.error("Failed to upload file");
      }
      setIsLoading(false);
  };

  return (
    <div className="Container1">
      <div className="container">
      {isLoading ? (
        <div style={{display: 'flex', justifyContent: "center", alignItems: "center", flexDirection: 'column'}}>
          <ReactLoading type="bubbles" color="#0000FF" height={100} width={50} />
          <p>Uploading your file please wait...</p>
        </div>
      ) : (
        <form onSubmit={handleSubmit}>
          <input type="file" onChange={handleFileChange} />
          <button type="submit">Upload</button>
        </form>
      )}
    </div>
    </div>
  );
}

export default FileUpload;