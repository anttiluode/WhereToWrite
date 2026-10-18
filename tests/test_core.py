import json
import tempfile
import unittest
from pathlib import Path

from where_to_write.memory import FactMemory
from where_to_write.probe import overlapping_positions, subject_span, PROMPTS, ARMS


class Tests(unittest.TestCase):
    def test_rewrite(self):
        m = FactMemory()
        self.assertIsNone(m.observe('Vera', 'yellow'))
        self.assertEqual(m.observe('Vera', 'red'), 'yellow')
        self.assertEqual(m.get('vera'), 'red')
        self.assertEqual(len(m.events), 2)

    def test_invalid(self):
        m = FactMemory()
        for name, color in [('', 'red'), ('A', 'purple'), ('A\nB', 'red')]:
            with self.assertRaises(ValueError):
                m.observe(name, color)

    def test_serialization(self):
        m=FactMemory();m.observe('Aino','green')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'facts.json';m.save(path)
            other=FactMemory();other.load(path)
            self.assertEqual(other.get('aino'),'green')
            path.write_text(json.dumps({'x':'purple'}))
            with self.assertRaises(ValueError):
                other.load(path)
            self.assertEqual(other.get('aino'),'green')

    def test_name_boundary(self):
        self.assertEqual(subject_span('Vera Vera? Answer:', 'Vera'), (5,9))
        with self.assertRaises(ValueError):
            subject_span('Veracity?', 'Vera')

    def test_unicode(self):
        self.assertEqual(subject_span('What color is Eero? Answer:', 'Eero'), (14,18))
        self.assertEqual(subject_span('What color is Aino? Answer:', 'Aino'), (14,18))

    def test_offsets(self):
        self.assertEqual(overlapping_positions([(0,4),(4,9),(10,14)], (6,12)),[1,2])
        self.assertEqual(overlapping_positions([(0,4),(5,10)], (4,5)),[])

    def test_prompts(self):
        self.assertEqual(len(ARMS),6)
        for t in PROMPTS:
            self.assertIn('{name}',t)
            self.assertIn('color',t)


if __name__=='__main__':
    unittest.main()
