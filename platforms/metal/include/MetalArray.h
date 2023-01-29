#ifndef OPENMM_OPENCLARRAY_H_
#define OPENMM_OPENCLARRAY_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2022 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "openmm/OpenMMException.h"
#include "openmm/common/windowsExportCommon.h"
#include "openmm/common/ArrayInterface.h"
#include "opencl.hpp"
#include <iostream>
#include <sstream>
#include <vector>

namespace OpenMM {

class MetalContext;

/**
 * This class encapsulates an Metal Buffer.  It provides a simplified API for working with it,
 * and for copying data to and from the Metal Buffer.
 */

class OPENMM_EXPORT_COMMON MetalArray : public ArrayInterface {
public:
    /**
     * Create an MetalArray object.  The object is allocated on the heap with the "new" operator.
     * The template argument is the data type of each array element.
     *
     * @param context           the context for which to create the array
     * @param size              the number of elements in the array
     * @param name              the name of the array
     * @param flags             the set of flags to specify when creating the Metal Buffer
     */
    template <class T>
    static MetalArray* create(MetalContext& context, size_t size, const std::string& name, cl_int flags = CL_MEM_READ_WRITE) {
        return new MetalArray(context, size, sizeof(T), name, flags);
    }
    /**
     * Create an MetalArray object that uses a preexisting Buffer.  The object is allocated on the heap with the "new" operator.
     * The template argument is the data type of each array element.
     *
     * @param context           the context for which to create the array
     * @param buffer            the Metal Buffer this object encapsulates
     * @param size              the number of elements in the array
     * @param name              the name of the array
     */
    template <class T>
    static MetalArray* create(MetalContext& context, cl::Buffer* buffer, size_t size, const std::string& name) {
        return new MetalArray(context, buffer, size, sizeof(T), name);
    }
    /**
     * Create an uninitialized MetalArray object.  It does not point to any Metal Buffer,
     * and cannot be used until initialize() is called on it.
     */
    MetalArray();
    /**
     * Create an MetalArray object.
     *
     * @param context           the context for which to create the array
     * @param size              the number of elements in the array
     * @param elementSize       the size of each element in bytes
     * @param name              the name of the array
     * @param flags             the set of flags to specify when creating the Metal Buffer
     */
    MetalArray(MetalContext& context, size_t size, int elementSize, const std::string& name, cl_int flags = CL_MEM_READ_WRITE);
    /**
     * Create an MetalArray object that uses a preexisting Buffer.
     *
     * @param context           the context for which to create the array
     * @param buffer            the Metal Buffer this object encapsulates
     * @param size              the number of elements in the array
     * @param elementSize       the size of each element in bytes
     * @param name              the name of the array
     */
    MetalArray(MetalContext& context, cl::Buffer* buffer, size_t size, int elementSize, const std::string& name);
    ~MetalArray();
    /**
     * Initialize this array.
     *
     * @param context           the context for which to create the array
     * @param size              the number of elements in the array
     * @param elementSize       the size of each element in bytes
     * @param name              the name of the array
     */
    void initialize(ComputeContext& context, size_t size, int elementSize, const std::string& name);
    /**
     * Initialize this object.
     *
     * @param context           the context for which to create the array
     * @param size              the number of elements in the array
     * @param elementSize       the size of each element in bytes
     * @param name              the name of the array
     * @param flags             the set of flags to specify when creating the Metal Buffer
     */
    void initialize(MetalContext& context, size_t size, int elementSize, const std::string& name, cl_int flags);
    /**
     * Initialize this object to use a preexisting Buffer.
     *
     * @param context           the context for which to create the array
     * @param buffer            the Metal Buffer this object encapsulates
     * @param size              the number of elements in the array
     * @param elementSize       the size of each element in bytes
     * @param name              the name of the array
     */
    void initialize(MetalContext& context, cl::Buffer* buffer, size_t size, int elementSize, const std::string& name);
    /**
     * Initialize this object.  The template argument is the data type of each array element.
     *
     * @param context           the context for which to create the array
     * @param size              the number of elements in the array
     * @param name              the name of the array
     * @param flags             the set of flags to specify when creating the Metal Buffer
     */
    template <class T>
    void initialize(MetalContext& context, size_t size, const std::string& name, cl_int flags = CL_MEM_READ_WRITE) {
        initialize(context, size, sizeof(T), name, flags);
    }
    /**
     * Initialize this object to use a preexisting Buffer.  The template argument
     * is the data type of each array element.
     *
     * @param context           the context for which to create the array
     * @param buffer            the Metal Buffer this object encapsulates
     * @param size              the number of elements in the array
     * @param name              the name of the array
     */
    template <class T>
    void initialize(MetalContext& context, cl::Buffer* buffer, size_t size, const std::string& name) {
        initialize(context, buffer, size, sizeof(T), name);
    }
    /**
     * Recreate the internal storage to have a different size.
     */
    void resize(size_t size);
    /**
     * Get whether this array has been initialized.
     */
    bool isInitialized() const {
        return (buffer != NULL);
    }
    /**
     * Get the size of the array.
     */
    size_t getSize() const {
        return size;
    }
    /**
     * Get the size of each element in bytes.
     */
    int getElementSize() const {
        return elementSize;
    }
    /**
     * Get the name of the array.
     */
    const std::string& getName() const {
        return name;
    }
    /**
     * Get the context this array belongs to.
     */
    ComputeContext& getContext();
    /**
     * Get the Metal Buffer object.
     */
    cl::Buffer& getDeviceBuffer() {
        return *buffer;
    }
    /**
     * Copy the values in a vector to the Buffer.
     */
    template <class T>
    void upload(const std::vector<T>& data, bool convert=false) {
        ArrayInterface::upload(data, convert);
    }
    /**
     * Copy the values in the Buffer to a vector.
     */
    template <class T>
    void download(std::vector<T>& data) const {
        ArrayInterface::download(data);
    }
    /**
     * Copy the values from host memory to the array.
     * 
     * @param data     the data to copy
     * @param blocking if true, this call will block until the transfer is complete.
     */
    void upload(const void* data, bool blocking=true) {
        uploadSubArray(data, 0, getSize(), blocking);
    }
    /**
     * Copy values from host memory to a subset of the array.
     * 
     * @param data     the data to copy
     * @param offset   the index of the element within the array at which the copy should begin
     * @param elements the number of elements to copy
     * @param blocking if true, this call will block until the transfer is complete.
     */
    void uploadSubArray(const void* data, int offset, int elements, bool blocking=true);
    /**
     * Copy the values in the Buffer to an array.
     * 
     * @param data     the array to copy the memory to
     * @param blocking if true, this call will block until the transfer is complete.
     */
    void download(void* data, bool blocking=true) const;
    /**
     * Copy the values in the Buffer to a second MetalArray.
     * 
     * @param dest     the destination array to copy to
     */
    void copyTo(ArrayInterface& dest) const;
private:
    MetalContext* context;
    cl::Buffer* buffer;
    size_t size;
    int elementSize;
    cl_int flags;
    bool ownsBuffer;
    std::string name;
};

} // namespace OpenMM

#endif /*OPENMM_OPENCLARRAY_H_*/
