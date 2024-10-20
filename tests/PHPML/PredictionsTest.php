<?php

namespace MLSandbox\PHPML;

use Phpml\Classification\SVC;
use Phpml\SupportVectorMachine\Kernel;
use PHPUnit\Framework\TestCase;

class PredictionsTest extends TestCase
{
    public function testPredictions(): void
    {
        $samples = [[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]];
        $labels = ['a', 'a', 'a', 'b', 'b', 'b'];

        $classifier = new SVC(Kernel::LINEAR, $cost = 1000);
        $classifier->train($samples, $labels);

        $result = $classifier->predict([3, 2]);
        $this->assertEquals('b', $result);

        $result = $classifier->predict([[3, 2], [1, 5]]);
        $this->assertEquals(['b', 'a'], $result);
    }
}
