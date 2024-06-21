const { execFile } = require('child_process');
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.set('view engine', 'ejs');

app.get('/', (req, res) => {
    res.render('index');
});

app.post('/upload', upload.single('image'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded.' });
    }

    const pythonScriptPath = path.join(__dirname, './utils/imageProcess.py');
    const imagePath = req.file.path;

    // Ensure the file exists
    if (!fs.existsSync(imagePath)) {
        return res.status(400).json({ error: 'Uploaded file not found.' });
    }

    execFile('python', [pythonScriptPath, imagePath], (error, stdout, stderr) => {
        if (error) {
            console.error(`Error: ${error.message}`);
            return res.status(500).json({ error: 'Server error' });
        }

        if (stderr) {
            console.error(`Stderr: ${stderr}`);
            return res.status(500).json({ error: 'Server error' });
        }

        let result;
        try {
            result = JSON.parse(stdout);
        } catch (parseError) {
            console.error(`JSON Parse Error: ${parseError.message}`);
            return res.status(500).json({ error: 'Error parsing response from Python script' });
        }

        if (result.error) {
            return res.status(400).json({ error: result.error });
        }

        res.json(result);
    });
});

app.listen(3000, () => {
    console.log('Server started on http://localhost:3000');
});
