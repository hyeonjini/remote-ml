module.exports = class Model {
    constructor(id, name, task, state, path, acc, loss, inference){
        this.id = id;
        this.name = name;
        this.task = task;
        this.state = state;
        this.path = path;
        this.acc = acc;
        this.loss = loss;
        this.inference = inference;
    }
};