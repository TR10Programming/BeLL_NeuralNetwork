<?php
//Input Neurons: $INin;
function m($arr1, $arr2) {
    $sum = 0;
    foreach($arr1 as $key => $val) {
        $sum += $val * $arr2[$key];
    }
    $sum = 1 / (1 + exp(-$sum));
    return $sum;
}

$INcmds;
?>