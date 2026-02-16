document.getElementById("uploadForm").addEventListener("submit", function (e) {
    e.preventDefault();

    let formData = new FormData(this);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerHTML = `
            <h3>Prediction Result</h3>
            <p><strong>Name:</strong> ${data.name}</p>
            <p><strong>Age:</strong> ${data.age}</p>
            <p><strong>Status:</strong> 
                <span style="color:${data.result === 'Tumor Detected' ? '#ef4444' : '#22c55e'}">
                    ${data.result}
                </span>
            </p>
            <p><strong>Confidence:</strong> ${data.confidence}%</p>
        `;
    })
    .catch(err => {
        document.getElementById("result").innerHTML =
            "<p style='color:red;'>Error while predicting</p>";
    });
});
