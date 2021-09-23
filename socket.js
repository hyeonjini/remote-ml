const SocketIO = require('socket.io');

module.exports = (server, app) =>{
    const io = SocketIO(server, { path: '/socket.io' });
    app.set('io', io);
    io.on('connection', (socket) => {
        const req = socket.request;
        const { headers: { referer } } = req;
        const modelId = referer.split('/')[referer.split('/').length - 1]
        socket.join(modelId);

        socket.on('disconnection', () => {
            socket.leave(modelId);
        });
    });
};