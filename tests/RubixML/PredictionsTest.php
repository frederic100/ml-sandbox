<?php

namespace MLSandbox\RubixML;

use PHPUnit\Framework\TestCase;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;

class PredictionsTest extends TestCase
{
    public function testPredictions(): void
    {
        $samples = [
            [3, 4, 50.5],
            [1, 5, 24.7],
            [4, 4, 62.0],
            [3, 2, 31.1],
        ];

        $labels = ['married', 'divorced', 'married', 'divorced'];

        $dataset = new Labeled($samples, $labels);

        $estimator = new KNearestNeighbors(3);

        $estimator->train($dataset);

        $validator = new HoldOut(0.2);

        $score = $validator->test($estimator, $dataset, new Accuracy());

        $this->assertGreaterThan(0.5, $score);
    }
}
