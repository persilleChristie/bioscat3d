#ifndef DIPOLE_H
#define DIPOLE_H

#include <Eigen/Dense>

/// @brief Class representing a dipole in 3D space.
/// @details This class encapsulates the properties of a dipole, including its position and direction.
/// The position is represented as a 3D vector, and the direction is a unit vector indicating the orientation of the dipole.
/// @note The direction vector is normalized to ensure it is a unit vector.
class Dipole {
public:
    /// @brief Default constructor for Dipole, initializes position and direction to zero.
    /// @details This constructor sets the position and direction of the dipole to zero vectors.
    Dipole(const Eigen::Vector3d& pos, const Eigen::Vector3d& dir)
        : position_(pos), direction_(dir.normalized()) {}

    /// @brief getPosition
    /// @details Returns the position of the dipole.
    const Eigen::Vector3d& getPosition() const { return position_; }
    /// @brief getDirection
    /// @details Returns the direction of the dipole as a unit vector.
    /// @note The direction vector is normalized to ensure it is a unit vector.
    const Eigen::Vector3d& getDirection() const { return direction_; }

    /// @brief setPosition
    /// @details Sets the position of the dipole to the specified vector. 
    void setPosition(const Eigen::Vector3d& pos) { position_ = pos; }
    /// @brief setDirection
    /// @details Sets the direction of the dipole to the specified vector, normalizing it to ensure it is a unit vector.
    /// @note Throws an exception if the provided direction vector is zero.
    void setDirection(const Eigen::Vector3d& dir) {
        if (dir.norm() == 0) {
            throw std::invalid_argument("Direction vector cannot be zero.");
        }
        direction_ = dir.normalized();
    }

private:
    Eigen::Vector3d position_;  // Dipole center
    Eigen::Vector3d direction_; // Dipole orientation (unit vector)
};

#endif // DIPOLE_H
