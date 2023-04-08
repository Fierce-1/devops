const router = require('express').Router();

router.route('/').get((req, res) => {
    res.status(401).json({ message: 'Invalid login credentials' })
});

module.exports = router;