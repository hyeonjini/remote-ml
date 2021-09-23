const fs = require('fs');
const path = require('path');
const targetFolder = path.resolve('ml/models');
const Model = require('../models/model');

const getModelList = () => {
    modelFolders = [];
    models = [];
    fs.readdirSync(targetFolder).forEach(file => {
        
        try{
            if(!file.startsWith('.')){
                const jsonFile = fs.readFileSync(path.join(targetFolder, file, "configure.json"), 'utf8');
                const jsonData = JSON.parse(jsonFile);
                const id = jsonData.config.id;
                const name = jsonData.config.name;
                const task = jsonData.config.task;
                const modelPath = path.join(targetFolder, file);
                const acc = jsonData.val_iter.acc[jsonData.val_iter.acc.length - 1];
                const loss = jsonData.val_iter.loss[jsonData.val_iter.loss.length - 1];
                var inference = path.join(targetFolder, file, "inference.json");
                if (fs.existsSync(inference))
                    models.push(new Model(id, name, task, "done", modelPath, acc, loss, true));
                else
                    models.push(new Model(id, name, task, "done", modelPath, acc, loss, false));
            }
        }
        catch (err) {
            models.push(new Model(file, "null", "null", "error", path.join(targetFolder, file), "null", "null", false));
        }
    });
    return models;
}

module.exports = {
    getModelList,
}