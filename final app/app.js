const express = require("express");
const app = express();

const PORT = 4000;

app.get("/", (req, res) => {
    res.send("Server is running successfully!");
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
