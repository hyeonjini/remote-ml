const express = require('express');
const ServerState = require('./serverState');
const parser = require('../utils/parser');
const readList = require('../utils/readList');
const state = ServerState.getInstance();
const router = express.Router();
const fs = require("fs");
const configure = require('../configure.json');

router.use((req, res, next) => {
    res.locals.model = null;
    res.locals.modelList = [];
    next();
})

router.get('/result', async (req, res) => {
    try{
        models = await readList.getModelList();
        res.render('result', {
            models,
        });
    }
    catch(err){
        console.error(err);
        next(err);
    }
});

router.get('/compare', (req, res) => {
    res.render('compare', {
        title: '모델 결과 비교',
    });
});

router.get('/', (req, res, next) => {
    try{
        res.render('main', {
            title: 'ui-ml',
            state,
            configure,
        });
    }
    catch(err){
        console.error(err);
        next(err);
    }
});
const requestCreate = async (query) => {
    var args = ['ml/train.py'];
    args = parser.createParameter(args, query);
    const { spawn, spawnSync } = require('child_process');
    const pyprocess = spawn('python', args);
    message = null;
    pyprocess.stdout.on('data', (data) => {
        console.log(data.toString());
        message = data.toString();
    });
    pyprocess.stderr.on('data', (data) => { //ml err 
        console.log(data.toString());
        message = data.toString();
    });

    return message;
} 
// api
router.get('/api/v1/requestCreate', (req, res) => {
    if (state.state){
        state.state = false;
        var args = ['ml/train.py'];
        args = parser.createParameter(args, req.query);
        // const { spawn } = require('child_process');
        // const pyprocess = spawn('python', args);
        console.log("python " + args.join(" "));
        const exec = require("child_process").exec;
        const pyprocess = exec("python " + args.join(" "));
        pyprocess.stderr.on('data', (data) => { //ml err 
            console.log(data.toString());
            state.state = true;
        });
    }else{
        console.log("ML module already is running");
    }
    return res.redirect('/');
});

router.post('/api/v1/:id/trainingProcess', (req, res) => {
    console.log(req.body);
    req.app.get('io').emit('update', req.body);
    return res.send('ok');
});

router.post('/api/v1/state/:id/training', (req, res) => {
    state.state = false; // false -> true
    console.log("ML module state:", state.state);
    console.log(req.body);
    req.app.get('io').emit('state', state.state);
    return res.send('ok');
});

router.post('/api/v1/state/:id/done', (req, res) => {
    
    state.state = true; // true -> false
    console.log("ML module state:", state.state);
    console.log(req.body);
    req.app.get('io').emit('state', state.state);
    return res.send('ok');
    
});

router.post('/temp', (req, res) => {
    console.log(req.body);
    return res.send('ok');
})
router.get('/api/v1/getModelList', async (req, res)=>{
    models = await readList.getModelList();
    console.log(models);
    return res.send(models);
});

router.get('/api/v1/:id/inference', async (req, res) => {

    console.log(req.params.id);
    const targetFolder = `ml/models/${req.params.id}/`;
    const jsonFile = fs.readFileSync(targetFolder + "inference.json");
    const jsonData = JSON.parse(jsonFile);

    test_image = jsonData.grad_cam.heatmap[0]
    console.log(test_image);

    res.sendFile(test_image);
})
router.get('/api/v1/:id/requestInference', async (req, res) => {

    if (state.state){
        state.state = false;
        var args = ['ml/inference.py', "--path", `ml/models/${req.params.id}`];
        console.log("python " + args.join(" "));
        const exec = require("child_process").exec;
        const pyprocess = exec("python " + args.join(" "));
        pyprocess.stderr.on('data', (data) => { //ml err 
            console.log(data.toString());
            state.state = true;
        });
    }else{
        console.log("ML module already is running");
    }
    return res.redirect('/result');
})

router.get('/api/v1/image/:id/:image', async (req, res) => {
    const targetPath = `/opt/ml/code/web_projects/ui_ml/ml/models/${req.params.id}/${req.params.image}`;
    console.log(targetPath);
    res.sendFile(targetPath);
})

router.get('/api/v1/get/currentState',(req, res) => {
    res.send(state.state);
});

router.get('/api/v1/:id/result', (req, res) => {

});

module.exports = router;