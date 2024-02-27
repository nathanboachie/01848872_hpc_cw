// ldc_test.cpp

#include "LidDrivenCavity.h"

#define BOOST_TEST_MODULE ldc_test
#include <boost/test/included/unit_test.hpp>


BOOST_AUTO_TEST_SUITE(LidDrivenCavityTest)

//Setter testing 
/*
BOOST_AUTO_TEST_CASE(SetterTesting)
{
    // Create object from class and set values 
    LidDrivenCavity test_object;
    test_object.SetDomainSize(5.0, 10.0);
    test_object.SetGridSize(10, 20);
    test_object.SetTimeStep(0.1);
    test_object.SetFinalTime(10.0);
    test_object.SetReynoldsNumber(100.0);

    // Check if the internal values are set correctly
    BOOST_CHECK_EQUAL(test_object.xlen(), 5.0);
    BOOST_CHECK_EQUAL(test_object.ylen(), 10.0);
    BOOST_CHECK_EQUAL(test_object.nx(), 10);
    BOOST_CHECK_EQUAL(test_object.ny(), 20);
    BOOST_CHECK_EQUAL(test_object.deltat(), 0.1);
    BOOST_CHECK_EQUAL(test_object.finalt(), 10.0);
    BOOST_CHECK_EQUAL(test_object.re(), 100.0);

}
*/

//Methods testing
BOOST_AUTO_TEST_CASE(methodstesting)
{
    LidDrivenCavity test_object;
    BOOST_CHECK_NO_THROW(test_object.Initialise());
    BOOST_CHECK_NO_THROW(test_object.Integrate());
    BOOST_CHECK_NO_THROW(test_object.WriteSolution("file"));
    BOOST_CHECK_NO_THROW(test_object.PrintConfiguration());
}

BOOST_AUTO_TEST_SUITE_END();
