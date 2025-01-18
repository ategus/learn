const express = require('express');
const { Pool } = require('pg');
const app = express();
const port = 3000;

require('dotenv').config();

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

app.use(express.static('public'));
app.use(express.json());

app.post('/add-value', async (req, res) => {
  const value = req.body.value;
  try {
    await pool.query('INSERT INTO random_values (timestamp, value) VALUES (NOW(), $1)', [value]);
    res.status(200).send('Value inserted');
  } catch (err) {
    console.error(err);
    res.status(500).send('Database error');
  }
});

app.listen(port, async () => {
  try {
    await pool.query(`
      CREATE TABLE IF NOT EXISTS random_values (
        time TIMESTAMPTZ NOT NULL,
        value DOUBLE PRECISION NOT NULL
      );
      SELECT create_hypertable('random_values', 'time', if_not_exists => TRUE);
    `);
    console.log(`Server running on http://localhost:${port}`);
  } catch (err) {
    console.error('Error initializing database:', err);
  }
});

