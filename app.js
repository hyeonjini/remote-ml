const express = require('express');
const path = require('path');
const nunjucks = require('nunjucks');
const morgan = require('morgan');
const webSocket = require('./socket');


const pageRouter = require('./routes/page');

const app = express();

app.set('port', process.env.PROT || 6006); //tensorboard port 임시 사용
app.set('host', process.env.HOST || 'localhost');
app.set('view engine', 'html');
nunjucks.configure('views', {
	express: app,
	watch: true,
});

app.use(morgan('dev'));
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

app.use('/', pageRouter);

app.use((req, res, next) => {
	const error = new Error(`${req.method} ${req.url} router does not exists.`);
	error.status = 404;
	next(error);
});

app.use((err, req, res, next) =>{
	res.locals.message = err.message;
	res.locals.error = process.env.NODE_ENV !== 'production' ? err : {};
	res.status(err.status || 500);
	res.render('500');
});

const server = app.listen(app.get('port'), app.get('host'), ()=>{
	console.log(app.get('port'), 'is opened');
});

webSocket(server, app);