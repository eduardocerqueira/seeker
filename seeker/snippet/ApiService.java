//date: 2023-05-30T16:40:28Z
//url: https://api.github.com/gists/1b4e340655fde4517ac769c270630146
//owner: https://api.github.com/users/asengsaragih

package com.suncode.relicbatik.api;

import com.suncode.relicbatik.model.ApiBaseResponse;
import com.suncode.relicbatik.model.Batik;
import com.suncode.relicbatik.model.ImageBatik;
import com.suncode.relicbatik.model.Store;
import com.suncode.relicbatik.model.StoreImage;
import com.suncode.relicbatik.model.TopBatikLike;
import com.suncode.relicbatik.model.User;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Field;
import retrofit2.http.FormUrlEncoded;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Query;

public interface ApiService {
    @FormUrlEncoded
    @POST("CreateOrUpdateClient")
    Call<User> createOrUpdateClient(
            @Field("name") String name,
            @Field("email") String email
    );

    @GET("GetTopBatikLike")
    Call<List<TopBatikLike>> getTopBatikLike();

    @GET("GetDashboardBatik")
    Call<List<Batik>> getDashboardBatik();

    @GET("GetBatiks")
    Call<List<Batik>> getBatiks(
            @Query("id") Integer id,
            @Query("name") String name,
            @Query("batik_class_id") Integer batik_class_id
    );

    @GET("GetImageBatik")
    Call<List<ImageBatik>> getImageBatik(
            @Query("batik_id") Integer batikId
    );

    @GET("GetLikeBatik")
    Call<List<Batik>> getLikeBatik(
            @Query("batik_id") Integer batikId,
            @Query("user_id") Integer userId
    );

    @FormUrlEncoded
    @POST("DeleteMappingUserBatikLike")
    Call<ApiBaseResponse> deleteMappingUserBatikLike(
            @Field("batik_id") Integer batikId,
            @Field("user_id") Integer user_id
    );

    @FormUrlEncoded
    @POST("InsertIntoMappingUserBatikLike")
    Call<ApiBaseResponse> insertIntoMappingUserBatikLike(
            @Field("batik_id") Integer batikId,
            @Field("user_id") Integer user_id
    );

    @GET("GetStoreWithBatikAndCurrentLocation")
    Call<List<Store>> getStoreWithBatikAndCurrentLocation(
            @Query("latitude") double latitude,
            @Query("longitude") double longitude,
            @Query("batik_id") int batikId
    );

    @GET("GetStoreWithBatikAndCurrentLocationAnotherData")
    Call<List<Store>> getStoreWithBatikAndCurrentLocationAnotherData(
            @Query("latitude") double latitude,
            @Query("longitude") double longitude,
            @Query("batik_id") int batikId,
            @Query("ids") String ids
    );

    @GET("GetSellerStoreItem")
    Call<List<Batik>> getSellerStoreItem(
            @Query("seller_store_id") int sellerStoreId
    );

    @GET("GetSellerStoreImage")
    Call<List<StoreImage>> getSellerStoreImage(
            @Query("seller_store_id") int sellerStoreId
    );
}
