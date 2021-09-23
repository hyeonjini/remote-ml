class PrivateSignleton {
    constructor() {
        this.state = true;
    }
}

class ServerState {
    constructor() {
        throw new Error('Use Singleton.getInstance()');
    }
    static getInstance() {
        if(!ServerState.instance){
            ServerState.instance = new PrivateSignleton();
        }
        return ServerState.instance;
    }
}

module.exports = ServerState;